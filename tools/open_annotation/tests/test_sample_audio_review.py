"""仅合成PCM／本机HTTP／SSE；不读真实私钥，不访问内部平台或GPU。"""
import base64
import contextlib
import hashlib
import http.server
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sample_audio_review', Path(__file__).resolve().parents[1]/'sample_audio_review.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def wav(samples=None):
    data = b''.join(struct.pack('<f', x) for x in (samples if samples is not None else [.1]*320))
    fmt = b'fmt ' + struct.pack('<IHHIIHH', 16, 3, 1, 16000, 64000, 4, 32)
    chunks = fmt + b'data' + struct.pack('<I', len(data)) + data
    return b'RIFF' + struct.pack('<I', len(chunks)+4) + b'WAVE' + chunks


def answer(**changes):
    obj = dict(summary='合成声音', heard_text='测试', language='Chinese', human_vocal_status='present',
               spoken_speech_status='present', first_audible_unit='测', last_audible_unit='试',
               start_cut_status='uncertain', end_cut_status='absent', adjacent_or_extra_speech_status='absent',
               start_evidence='合成例无法确定', end_evidence='没有可辨截断', extra_speech_evidence='未检出',
               uncertainties=['不作人工真值'])
    obj.update(changes)
    return obj


def sse(obj=None, *, done=True, finish='stop', model='qwen3.8-omni-flash', malformed=False):
    content = json.dumps(answer() if obj is None else obj, ensure_ascii=False)
    chunks = [dict(model=model, choices=[dict(index=0, delta={'reasoning_content':'单独保留的推理', 'content':content}, finish_reason=None)]),
              dict(model=model, choices=[dict(index=0, delta={}, finish_reason=finish)]),
              dict(model=model, choices=[], usage={'completion_tokens':12, 'completion_tokens_details':{'reasoning_tokens':2}})]
    raw = b''.join(b'data: ' + json.dumps(c, ensure_ascii=False).encode() + b'\r\n\r\n' for c in chunks)
    if malformed:
        raw = b'data: {invalid}\n\n'
    return raw + (b'data: [DONE]\n\n' if done else b'')


@contextlib.contextmanager
def server(responses):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            with lock:
                index = len(requests)
                requests.append({'body':json.loads(body), 'headers':dict(self.headers), 'started':time.monotonic()})
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            entry = responses[min(index, len(responses)-1)]
            try:
                time.sleep(entry.get('delay', 0))
                self.send_response(entry.get('status', 200))
                self.send_header('Content-Type', entry.get('content_type', 'text/event-stream'))
                for name,value in entry.get('headers',{}).items():
                    self.send_header(name,value)
                raw = entry.get('raw', sse())
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                if entry.get('drip_delay'):
                    for part in raw:
                        self.wfile.write(bytes([part]));self.wfile.flush()
                        time.sleep(entry['drip_delay'])
                else:self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock:
                    active[0] -= 1
                    requests[index]['finished'] = time.monotonic()
    lock = threading.Lock()
    requests, active, peak = [], [0], [0]
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield 'http://127.0.0.1:%d/v1' % httpd.server_port, requests, peak
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(3)
        if thread.is_alive():
            raise RuntimeError('fixture HTTP server did not stop')


class SampleAudioReviewTests(unittest.TestCase):
    def fixture(self, root, count=3):
        cases = []
        for i in range(count):
            path = root/('input%d.wav'%i)
            raw = wav([.1]*(320+i))
            path.write_bytes(raw)
            frames, pcm = M.pcm_identity(raw)
            cases.append(dict(case_id='case%d'%i, source_id='src_'+M.sha(('source%d'%i).encode()),wavpath=str(path), file_sha256=M.sha(raw),
                              pcm_float32le_sha256=pcm, frames=frames, sr=16000,
                              reference_text='参考可能错误，不是指令'))
        path = root/'manifest.json'
        path.write_bytes(M.canonical({'cases':cases,'source_ids':[x['source_id']for x in cases]}))
        key = root/'private-key'
        key.write_text('synthetic-private-credential')
        key.chmod(0o600)
        return path, M.sha(path.read_bytes()), key

    def execute(self, root, fixture, base, **kwargs):
        return M.run(fixture[0], fixture[1], root/'out', key_file=fixture[2], execute=True,
                     purpose='cpu_fixture', network_base=base, **kwargs)

    def test_default_is_audio_precheck_without_key_network_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); f=self.fixture(root)
            with patch.object(M, 'load_key', side_effect=AssertionError('must not read key')):
                result=M.run(f[0], f[1], root/'out')
            self.assertEqual(result['API_calls'], 0)
            self.assertFalse(result['credential_read'])
            self.assertFalse((root/'out').exists())

    def test_exact_pcm_sha_frames_and_nonfinite_truncation_refused(self):
        for raw in (wav([float('nan')]), wav([float('inf')]), wav()[:-1], b'not-wave', wav()+b'extra'):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                M.pcm_identity(raw)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); f=self.fixture(root,1)
            data=json.loads(f[0].read_bytes()); data['cases'][0]['frames']=True
            f[0].write_bytes(M.canonical(data))
            with self.assertRaises(ValueError):
                M.run(f[0],M.sha(f[0].read_bytes()),root/'out')
            with self.assertRaises(ValueError):
                M.run(f[0],'f'*64,root/'out')

    def test_source_mismatch_duplicate_id_and_reference_missing_refused(self):
        for kind in ('sha','duplicate','reference'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();f=self.fixture(root,2);data=json.loads(f[0].read_bytes())
                if kind=='sha':data['cases'][0]['pcm_float32le_sha256']='f'*64
                if kind=='duplicate':data['cases'][1]['case_id']=data['cases'][0]['case_id']
                if kind=='reference':data['cases'][0].pop('reference_text')
                f[0].write_bytes(M.canonical(data))
                with self.assertRaises(ValueError):
                    M.run(f[0],M.sha(f[0].read_bytes()),root/'out',profile='reference' if kind=='reference' else 'blind')

    def test_canary_then_bounded_fanout_preserves_SSE_usage_model_and_no_secret(self):
        with tempfile.TemporaryDirectory() as tmp, server([{'delay':.05}]) as (base,requests,peak):
            root=Path(tmp).resolve();f=self.fixture(root,5)
            result=self.execute(root,f,base)
            self.assertEqual(result['API_attempts'],5)
            self.assertEqual(result['schema_usable_count'],5)
            self.assertLessEqual(peak[0],2)
            self.assertTrue(all(x['started']>=requests[0]['finished'] for x in requests[1:]))
            self.assertTrue(all(x['body']['reasoning_effort']=='none' for x in requests))
            self.assertTrue(all(x['headers']['Source-Sn']=='zhonglimin-test' for x in requests))
            self.assertTrue(all('参考可能错误' not in json.dumps(x['body'],ensure_ascii=False) for x in requests))
            self.assertFalse(result['human_accuracy_certified'])
            record=json.loads((root/'out/records/case0.json').read_bytes());a=record['attempts'][0]
            self.assertEqual(a['returned_models'],['qwen3.8-omni-flash'])
            self.assertEqual(a['finish_reasons'],['stop'])
            self.assertTrue(a['stream_done']);self.assertIn('单独保留',a['reasoning_text'])
            self.assertEqual(a['parsed']['heard_text'],'测试');self.assertTrue(a['usage'])
            for path in (root/'out').rglob('*'):
                if path.is_file():self.assertNotIn(f[2].read_bytes(),path.read_bytes())

    def test_reference_is_data_and_conflicting_match_preserved_without_human_pass(self):
        obj=answer(reference_text_match='matches',reference_text_issues=[])
        with tempfile.TemporaryDirectory() as tmp,server([{'raw':sse(obj)}]) as (base,requests,_):
            root=Path(tmp).resolve();f=self.fixture(root,2);result=self.execute(root,f,base,profile='reference')
            self.assertEqual(len(requests),2);self.assertFalse(result['stop_fanout'])
            prompt=requests[0]['body']['messages'][0]['content'][0]['text']
            self.assertIn('参考数据',prompt);self.assertIn('参考可能错误',prompt)
            example=prompt.split('参考数据：')[0]
            self.assertIn('"reference_text_match"',example)
            self.assertIn('"reference_text_issues"',example)
            record=json.loads((root/'out/records/case0.json').read_bytes())
            self.assertIn('reference_match_conflicts_with_heard_text',record['attempts'][0]['content_flags'])
            self.assertFalse(record['usable_for_training'])

    def test_canary_finish_done_model_or_schema_error_stops_fanout(self):
        variants=[sse(done=False),sse(finish='length'),sse(model='unexpected-backend'),sse(malformed=True),sse({'heard_text':'缺字段'})]
        for raw in variants:
            with self.subTest(raw=raw[:20]),tempfile.TemporaryDirectory() as tmp,server([{'raw':raw}]) as (base,requests,_):
                root=Path(tmp).resolve();f=self.fixture(root,5);result=self.execute(root,f,base)
                self.assertTrue(result['stop_fanout']);self.assertEqual(len(requests),1)
                self.assertEqual(sum(x['status']=='not_dispatched_after_stop'for x in result['cases']),4)

    def test_deterministic_gateway579_preserved_and_not_retried_for_every_source(self):
        with tempfile.TemporaryDirectory() as tmp,server([{'status':579,'raw':b'source-sn-lack'}]) as (base,requests,_):
            root=Path(tmp).resolve();f=self.fixture(root,8);result=self.execute(root,f,base)
            self.assertTrue(result['stop_fanout']);self.assertEqual(len(requests),1)
            a=json.loads((root/'out/records/case0.json').read_bytes())['attempts'][0]
            self.assertEqual(a['http_status'],579);self.assertEqual(a['error_body'],'source-sn-lack')
            self.assertFalse(a['stream_done']);self.assertEqual(a['returned_models'],[])
            self.assertEqual((root/'out/raw/case0.attempt-1.http-error.bin').read_bytes(),b'source-sn-lack')

    def test_retry_transient_http_then_reuses_successful_canary_once(self):
        with tempfile.TemporaryDirectory() as tmp,server([{'status':503,'raw':b'temporary'},{'raw':sse()}]) as (base,requests,_):
            root=Path(tmp).resolve();f=self.fixture(root,2);result=self.execute(root,f,base)
            self.assertFalse(result['stop_fanout']);self.assertEqual(len(requests),3)
            a=json.loads((root/'out/records/case0.json').read_bytes())['attempts']
            self.assertEqual(len(a),2);self.assertEqual(a[0]['http_status'],503);self.assertEqual(a[1]['http_status'],200)

    def test_private_key_mode_and_production_alternate_destination_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();f=self.fixture(root,1);f[2].chmod(0o644)
            with self.assertRaises(ValueError):M.load_key(f[2])
            with self.assertRaises(ValueError):M.run(f[0],f[1],root/'out',network_base='http://127.0.0.1:1/v1')
            with self.assertRaises(ValueError):M.run(f[0],f[1],root/'out',workers=3)

    def test_error_echo_secret_redacted_and_existing_output_never_overwritten(self):
        body=b'bad synthetic-private-credential'
        with tempfile.TemporaryDirectory() as tmp,server([{'status':400,'raw':body}]) as (base,_,_):
            root=Path(tmp).resolve();f=self.fixture(root,1);self.execute(root,f,base)
            before={str(p):p.read_bytes()for p in (root/'out').rglob('*')if p.is_file()}
            self.assertTrue(all(f[2].read_bytes()not in x for x in before.values()))
            with self.assertRaises(FileExistsError):self.execute(root,f,base)
            self.assertEqual(before,{str(p):p.read_bytes()for p in (root/'out').rglob('*')if p.is_file()})

    def test_json_duplicates_nonfinite_and_placeholders_not_invented_words(self):
        for text in ('{"a":1,"a":2}','{"x":NaN}','{"x":1e999}'):
            with self.assertRaises(ValueError):M.strict_json(text)
        errors,content=M.output_flags(answer(heard_text='[听不清]',spoken_speech_status='absent'),'blind',{})
        self.assertFalse(errors);self.assertNotIn('heard_text_with_absent_speech_needs_review',content)

    def test_later_protocol_failure_stops_new_dispatch_but_preserves_inflight(self):
        responses=[{'raw':sse()},{'status':579,'raw':b'source-sn-lack'},{'raw':sse(),'delay':.05}]
        with tempfile.TemporaryDirectory() as tmp,server(responses) as (base,requests,peak):
            root=Path(tmp).resolve();f=self.fixture(root,12);result=self.execute(root,f,base)
            self.assertTrue(result['stop_fanout']);self.assertLessEqual(len(requests),3);self.assertLessEqual(peak[0],2)
            self.assertEqual(len(result['cases']),12)
            self.assertTrue((root/'out/summary.json').is_file())

    def test_redirect_not_followed_with_credentials(self):
        with tempfile.TemporaryDirectory()as tmp,server([{'raw':sse()}])as (target,target_requests,_):
            with server([{'status':302,'headers':{'Location':target+'/chat/completions'},'raw':b'redirect'}])as (base,requests,_):
                root=Path(tmp).resolve();f=self.fixture(root,3);result=self.execute(root,f,base)
                self.assertTrue(result['stop_fanout']);self.assertEqual(len(requests),1)
                self.assertEqual(target_requests,[])

    def test_SSE_secret_echo_is_redacted_and_flagged(self):
        obj=answer(heard_text='synthetic-private-credential')
        with tempfile.TemporaryDirectory()as tmp,server([{'raw':sse(obj)}])as (base,requests,_):
            root=Path(tmp).resolve();f=self.fixture(root,3);result=self.execute(root,f,base)
            self.assertTrue(result['stop_fanout']);self.assertEqual(len(requests),1)
            record=json.loads((root/'out/records/case0.json').read_bytes())
            self.assertIn('credential_echo_redacted',record['attempts'][0]['protocol_flags'])
            self.assertTrue(all(f[2].read_bytes()not in p.read_bytes()for p in (root/'out').rglob('*')if p.is_file()))

    def test_production_source_roster24_limit_and_exact_crop_membership(self):
        for kind in ('25sources','missing_roster','extra_roster','unknown_case_source','duplicate_roster'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp).resolve();f=self.fixture(root,25 if kind=='25sources' else 2);d=json.loads(f[0].read_bytes())
                if kind=='missing_roster':d.pop('source_ids')
                if kind=='extra_roster':d['source_ids'].append('src_'+'f'*64)
                if kind=='unknown_case_source':d['cases'][0]['source_id']='src_'+'f'*64
                if kind=='duplicate_roster':d['source_ids'].append(d['source_ids'][0])
                f[0].write_bytes(M.canonical(d))
                with self.assertRaises(ValueError):M.run(f[0],M.sha(f[0].read_bytes()),root/'out')
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp).resolve();f=self.fixture(root,32);d=json.loads(f[0].read_bytes())
            d['source_ids']=d['source_ids'][:24]
            for i,c in enumerate(d['cases']):c['source_id']=d['source_ids'][i%24]
            f[0].write_bytes(M.canonical(d));result=M.run(f[0],M.sha(f[0].read_bytes()),root/'out')
            self.assertEqual(result['original_source_count'],24);self.assertEqual(result['cases'],32)

    def test_read1_dripping_SSE_and_HTTP_error_body_cannot_extend_deadline(self):
        for status,raw in [(200,b'data: '+b'x'*2000),(503,b'x'*2000)]:
            with self.subTest(status=status),tempfile.TemporaryDirectory()as tmp,server([{'status':status,'raw':raw,'drip_delay':.005}])as (base,requests,_):
                root=Path(tmp).resolve();f=self.fixture(root,3)
                with patch.object(M,'WALL_BUDGET',.07),patch.object(M,'TIMEOUT',.5):
                    result=self.execute(root,f,base)
                self.assertTrue(result['stop_fanout']);self.assertLessEqual(len(requests),2)
                a=json.loads((root/'out/records/case0.json').read_bytes())['attempts']
                self.assertTrue(all(x['seconds']<.3 for x in a))
                self.assertTrue(all(not x['stream_done']for x in a))
                self.assertTrue(all(x['status']=='transport_failed'for x in a))

    def test_late_DONE_rejected_and_JSON_escaped_credential_keys_values_redacted(self):
        class Late:
            def read1(self,n):time.sleep(.04);return b'data: [DONE]\n\n'
        class FakeSocket:
            def settimeout(self,t):pass
        class Raw:pass
        value=Late();value.fp=Raw();value.fp.raw=Raw();value.fp.raw._sock=FakeSocket()
        import io
        with patch.object(M.socket,'socket',FakeSocket),patch.object(M,'WALL_BUDGET',.01):
            result,failure=M.stream_response(value,io.BytesIO(),'fake-key',time.monotonic())
        self.assertIsInstance(failure,TimeoutError);self.assertFalse(result['stream_done'])
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'safe.json';key='synthetic-"quoted\\credential'
            M.write_json(p,{key:{'value':key}},key)
            saved=json.loads(p.read_bytes())
            self.assertFalse(M.contains_key(saved,key));self.assertNotIn(key.encode(),p.read_bytes())

    def test_credential_reassembled_across_reasoning_frames_flags_and_redacts_raw(self):
        text=json.dumps(answer(),ensure_ascii=False)
        values=[dict(model='qwen3.8-omni-flash',choices=[dict(index=0,delta={'content':text[:30],'reasoning_content':'synthetic-private-'},finish_reason=None)]),
                dict(model='qwen3.8-omni-flash',choices=[dict(index=0,delta={'content':text[30:],'reasoning_content':'credential'},finish_reason='stop')])]
        raw=b''.join(b'data: '+json.dumps(v,ensure_ascii=False).encode()+b'\n\n'for v in values)+b'data: [DONE]\n\n'
        with tempfile.TemporaryDirectory()as tmp,server([{'raw':raw}])as (base,requests,_):
            root=Path(tmp).resolve();f=self.fixture(root,2);result=self.execute(root,f,base)
            self.assertTrue(result['stop_fanout']);self.assertEqual(len(requests),1)
            a=json.loads((root/'out/records/case0.json').read_bytes())['attempts'][0]
            self.assertIn('credential_echo_redacted',a['protocol_flags'])
            self.assertTrue(a['raw_SSE_entire_response_redacted']);self.assertEqual(a['chunks'],[])
            self.assertNotIn(b'synthetic-private-',(root/'out/raw/case0.attempt-1.sse').read_bytes())

    def test_HTTP_error_response_closed_after_deadline_or_bounded_partial_body(self):
        # 错误体没读到EOF时HTTPError不会自己归还响应，必须在重试/返回前显式关闭。
        variants=[{'status':503,'raw':b'x'*2000,'drip_delay':.005},
                  {'status':400,'raw':b'x'*131072}]
        for entry in variants:
            with self.subTest(status=entry['status']),tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp).resolve();f=self.fixture(root,1);seen=[]
                real_body=M.error_body
                def capture(response,started):
                    seen.append(response)
                    return real_body(response,started)
                try:
                    with server([entry])as (base,requests,_):
                        with patch.object(M,'error_body',side_effect=capture),patch.object(M,'WALL_BUDGET',.07),patch.object(M,'TIMEOUT',.5):
                            result=self.execute(root,f,base)
                    self.assertTrue(result['stop_fanout']);self.assertTrue(seen)
                    self.assertTrue(all(e.fp.isclosed()for e in seen),'HTTPError response left open')
                    if entry['status']==400:
                        a=json.loads((root/'out/records/case0.json').read_bytes())['attempts'][0]
                        self.assertTrue(a['error_body_may_be_truncated'])
                        self.assertEqual(len((root/'out/raw/case0.attempt-1.http-error.bin').read_bytes()),65536)
                finally:
                    # 旧实现的反例也及时关本测试创建的响应，不把测试本身变成泄漏来源。
                    for response in seen:response.close()


if __name__ == '__main__':
    unittest.main()
