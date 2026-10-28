"""桥接工具的有界原件、有效EOS及目标变化CPU反例；不读真实媒体。"""
import copy
import hashlib
import json
import struct
from pathlib import Path
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import compare_native_b1_bridge as B


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        source={'source_id':'synthetic','dataset':'voxceleb2','sidecar_relpath':'voxceleb2/fake.wav.annotation.json'}
        window=dict(window_id='w0',input_pcm_sha256='a'*64,model_input_frames=16000,model_sample_rate=16000,
            context_start_sample=0,context_end_sample=16000,owned_start_sample=0,owned_end_sample=16000,
            resampling_policy='synthetic',time_mapping='same',asr=dict(status='completed',raw_text='hello',normalized_text='hello',
                normalization_audit={},language='English',native=dict(generated_token_ids=[10,2],eos_observed=True,first_eos_index=1,max_new_tokens=512)),
            alignment=dict(target_text='hello',items=[dict(text='hello',start_time=.1,end_time=.2)],native_timestamps_ms=[100,200],fixed_timestamps_ms=[100,200]))
        record=dict(source=source,source_identity=dict(decode_status='decoded_complete',file_sha256='b'*64,pcm_sha256='c'*64,
            pcm_hash_policy='synthetic',sample_rate=16000,frames=16000,channels=1),windows=[window],errors=[],vad={},units=[],decision={},quality_axes={})
        c=dict(models={},source_files={},alignment={},decoder={},quality_policy={},version='old',purpose='pilot_validation',
            runtime=dict(asr_compute={'dtype':'float32','attention':'sdpa'},max_batch=1,execution_policy='single_window_independent',max_new_tokens=512))
        self.old=dict(contract=c,contract_ref={'path':'/old','sha256':'1'*64},index_ref={},track_ref={},parent_sha256='2'*64,
            records={'synthetic':record},refs={})
        self.new=copy.deepcopy(self.old);self.new['contract']['version']='new'
        self.new['contract']['runtime'].update(execution_policy='asr_fixed_cohort_fa_single/v1',asr_batching=dict(
            policy='fixed_cohort_until_durable/v1',max_requests=1,ordering='frames_then_row_id_within_source_group/v1'))
    def test_same_math_is_not_same_code_speed_or_human_truth(self):
        result=B.compare(self.old,self.new);self.assertEqual(result['counts']['effective_tokens_changed'],0)
        self.assertEqual(result['rows'][0]['windows'][0]['maximum_endpoint_delta_ms']['native_timestamps_ms'],0)
        self.assertIsNone(result['speedup_ratio']);self.assertFalse(result['formal_admission']);self.assertFalse(result['human_accuracy_certified'])
    def test_padding_after_eos_does_not_become_effective_token_change(self):
        n=self.new['records']['synthetic']['windows'][0]['asr']['native'];n['generated_token_ids']=[10,2,2];n['effective_generated_token_ids']=[10,2]
        counts=B.compare(self.old,self.new)['counts'];self.assertEqual(counts['raw_tokens_changed'],1);self.assertEqual(counts['effective_tokens_changed'],0)
    def test_different_target_never_receives_zero_endpoint_delta(self):
        w=self.new['records']['synthetic']['windows'][0];w['asr'].update(raw_text='world',normalized_text='world');w['alignment']['target_text']='world'
        result=B.compare(self.old,self.new);self.assertEqual(result['counts']['text_changed'],1)
        self.assertEqual(result['rows'][0]['windows'][0]['maximum_endpoint_delta_ms'],{})
    def test_generation_control_and_source_union_changes_refused(self):
        self.new['contract']['runtime']['max_new_tokens']=16
        with self.assertRaisesRegex(ValueError,'generation'):B.compare(self.old,self.new)
        self.new['contract']['runtime']['max_new_tokens']=512;self.new['records']={}
        with self.assertRaisesRegex(ValueError,'source sets'):B.compare(self.old,self.new)
    def test_eos_corruption_cannot_pass_bridge(self):
        w=self.new['records']['synthetic']['windows'][0];w['asr']['native']['first_eos_index']=True
        with self.assertRaisesRegex(ValueError,'EOS index'):B.compare(self.old,self.new)
    def test_native_actual_eos_suffix_and_effective_denominator_refused(self):
        n=self.new['records']['synthetic']['windows'][0]['asr']['native']
        n.update(padding_policy={},eos_token_ids=[2],padding_token_id=0,generated_token_ids=[10,2,0],
            effective_generated_token_ids=[10,2],post_eos_token_ids=[0],effective_generated_token_count=2,raw_returned_token_count=3)
        B.compare(self.old,self.new)
        n['generated_token_ids']=[10,2,99];n['post_eos_token_ids']=[99]
        with self.assertRaisesRegex(ValueError,'suffix'):B.compare(self.old,self.new)
    def test_torn_completion_and_wrong_contract_refused(self):
        shard=dict(id='s0',sha256='a'*64,count=1);index=dict(shards=[shard])
        body=B.canonical(dict(shard_id='s0',shard_sha256='a'*64,source_count=1,contract_sha256='b'*64,index_sha256='c'*64))
        raw=b'RDMAP001'+struct.pack('>I',len(body))+hashlib.sha256(body).digest()+body+bytes(980-len(body))
        B.completion_slots(raw,index,'b'*64,'c'*64)
        with self.assertRaisesRegex(ValueError,'identity'):B.completion_slots(raw,index,'d'*64,'c'*64)
        with self.assertRaisesRegex(ValueError,'checksum'):B.completion_slots(raw[:50]+bytes([raw[50]^1])+raw[51:],index,'b'*64,'c'*64)
    def test_nonfinite_duplicate_json_and_wrong_or_symlink_pin_refused(self):
        for raw in [b'{"a":NaN}',b'{"a":1,"a":2}']:
            with self.assertRaises(ValueError):B.json_value(raw)
        p=self.root/'input.json';p.write_bytes(b'{}\n')
        with self.assertRaisesRegex(ValueError,'SHA'):B.pin({'path':str(p),'sha256':'0'*64})
        alias=self.root/'alias.json';alias.symlink_to(p)
        with self.assertRaisesRegex(ValueError,'physical'):B.read(alias)
    def test_actual_small_files_parent_commit_and_bounded_prepared_readback(self):
        def put(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);raw=B.canonical(value);path.write_bytes(raw)
            return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
        for native in (False,True):
            label='native'if native else'old';root=self.root/label;trackdata=self.new if native else self.old
            contract=trackdata['contract'];cp=put(root/'contract.json',contract)
            source=copy.deepcopy(trackdata['records']['synthetic']['source']);raw=B.canonical(source);shard=root/'input/s.jsonl';shard.parent.mkdir(parents=True);shard.write_bytes(raw)
            index=put(root/'input/index.json',dict(source_count=1,shards=[dict(id='s0',path='s.jsonl',sha256=hashlib.sha256(raw).hexdigest(),count=1)]))
            record=copy.deepcopy(trackdata['records']['synthetic']);job=B.digest(dict(source_id='synthetic',source_record_sha256=B.digest(source),contract_sha256=cp['sha256']))
            record.update(contract={'sha256':cp['sha256']},execution={'job_key':job},provenance=dict(runtime=contract['runtime'],models={},code_sha256='9'*64))
            record['decision']={'state':'unknown'}
            sp=put(root/'labels'/contract['version']/source['sidecar_relpath'],record);sr=Path(sp['path']).read_bytes()
            marker=put(Path(sp['path']+'.commit.json'),dict(schema='redimnet2-sidecar-commit/v1',source_id='synthetic',job_key=job,
                contract_sha256=cp['sha256'],source_record_sha256=hashlib.sha256(raw).hexdigest(),sidecar_sha256=sp['sha256'],bytes=len(sr)))
            chunk=put(root/'queue/completed/s0.json',dict(schema='redimnet2-chunk-commit/v1',shard_id='s0',shard_sha256=hashlib.sha256(raw).hexdigest(),
                contract_sha256=cp['sha256'],index_sha256=index['sha256'],source_count=1,sources=[dict(source_id='synthetic',job_key=job,
                sidecar_sha256=sp['sha256'],sidecar_bytes=len(sr),commit_sha256=marker['sha256'],decision='unknown')]))
            slot=dict(shard_id='s0',shard_sha256=hashlib.sha256(raw).hexdigest(),source_count=1,contract_sha256=cp['sha256'],index_sha256=index['sha256'],
                source_ids_sha256=B.digest(['synthetic']),commit_sha256=chunk['sha256'],decision_counts=dict(candidate=0,unknown=1,failed=0))
            body=B.canonical(slot);(root/'queue/completion_map.bin').write_bytes(b'RDMAP001'+struct.pack('>I',len(body))+hashlib.sha256(body).digest()+body+bytes(980-len(body)))
            summary=dict(worker_id='w0',contract_sha256=cp['sha256'],index_sha256=index['sha256'],chunk_receipt_refs=[dict(chunk,id='s0',source_count=1)])
            summaryref=put(root/'resources/w0/worker_summary.json',summary)
            readyref=put(root/'resources/w0/terminal_ready.json',dict(summary_path=summaryref['path'],summary_sha256=summaryref['sha256'],
                contract_sha256=cp['sha256'],index_sha256=index['sha256'],bundle_sha256='9'*64,nonce='fixture'))
            put(root/'resources/w0/terminal_ack.json',dict(nonce='fixture',ready_sha256=readyref['sha256']))
            parent=put(root/'resources/parent_result.json',dict(contract_sha256=cp['sha256'],index_sha256=index['sha256'],bundle_sha256='9'*64,
                resource_returned=True,full_sidecar_readback=True,verification={'passed':True,'database_created':False},worker_count=1,
                workers=[dict(worker_id='w0',terminal=dict(summary=summary,summary_sha256=summaryref['sha256'],ready_sha256=readyref['sha256']))]))
            track=put(root/'track.json',dict(contract=cp,index=index,output_root=str(root/'labels'),resources_root=str(root/'resources'),queue_root=str(root/'queue')))
            profile='native-fp32-sdpa-b1'if native else'asr-fp32-sdpa'
            prepared=put(root/'prepared.json',dict(schema='annotation-native-batch-prepared/v1'if native else'annotation-bounded-speed-prepared/v1',
                source_count=1,index=index,variants={profile:dict(track=track,freeze={'bundle_sha256':'9'*64})}))
            loaded=B.load(prepared,profile,native=native,parent_sha=parent['sha256']);self.assertEqual(set(loaded['records']),{'synthetic'})
            changed=copy.deepcopy(record);changed['windows'][0]['asr']['raw_text']='forged-self-consistent-pair'
            new_sp=put(Path(sp['path']),changed)
            descriptor=json.loads(Path(marker['path']).read_text());descriptor.update(sidecar_sha256=new_sp['sha256'],bytes=Path(sp['path']).stat().st_size)
            put(Path(marker['path']),descriptor)
            with self.assertRaisesRegex(ValueError,'SHA'):B.load(prepared,profile,native=native,parent_sha=parent['sha256'])
        broken=json.loads(Path(prepared['path']).read_text());broken['source_count']=65;prepared=put(Path(prepared['path']),broken)
        with self.assertRaisesRegex(ValueError,'64'):B.load(prepared,profile,native=True,parent_sha=parent['sha256'])


if __name__=='__main__':unittest.main()
