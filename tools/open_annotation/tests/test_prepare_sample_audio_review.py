"""只用合成音频验证有限采集/视图身份守卫，不连接服务器或模型。"""
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
P=load('prepare_sample_audio_review');C=load('collect_sample_audio')
sha=lambda b:hashlib.sha256(b).hexdigest()
class Preparation(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.audio=self.root/'audio';(self.audio/'q01').mkdir(parents=True)
        self.wave=np.sin(np.arange(8000,dtype=np.float32)*.03).astype('<f4');self.path=self.audio/'q01/source.wav';sf.write(self.path,self.wave,16000,subtype='FLOAT')
        self.side=self.root/'side.json';self.record=dict(source={'source_id':'src_fixture'},source_identity=dict(frames=8000,channels=1,sample_rate=16000,file_sha256=sha(self.path.read_bytes()),pcm_sha256=sha(self.wave.tobytes())),windows=[dict(window_id='w0',context_start_sample=0,context_end_sample=8000,owned_start_sample=0,owned_end_sample=8000,input_pcm_sha256=sha(self.wave.tobytes()),asr={'raw_text':'合成参考'})]);self.items=[dict(review_id='q01',local_sidecar=str(self.side),row=dict(source_id='src_fixture',audio_path='/fixed/source.wav',audio_recorded_file_sha256=sha(self.path.read_bytes()),sidecar_sha256=''))];self.sel=self.root/'selection.json';self.save()
    def save(self):
        self.side.write_text(json.dumps(self.record));self.items[0]['row']['sidecar_sha256']=sha(self.side.read_bytes());self.sel.write_text(json.dumps(self.items))
    def run_prepare(self):return P.prepare(self.sel,self.audio,self.root/'output')
    def test_exact_pcm_and_source_binding(self):
        self.run_prepare();m=json.loads((self.root/'output/manifest.json').read_text());self.assertEqual(m['source_ids'],['src_fixture']);self.assertEqual(m['cases'][0]['source_input_sample_bounds'],[0,8000]);self.assertFalse(m['cases'][0]['artificial_context_cut_right']);self.assertEqual(m['cases'][0]['pcm_float32le_sha256'],sha(self.wave.tobytes()))
    def test_encoded_mismatch(self):
        self.items[0]['row']['audio_recorded_file_sha256']='0'*64;self.save()
        with self.assertRaisesRegex(ValueError,'encoded'):self.run_prepare()
    def test_pcm_mismatch(self):
        self.record['source_identity']['pcm_sha256']='0'*64;self.save()
        with self.assertRaisesRegex(ValueError,'PCM SHA'):self.run_prepare()
    def test_model_context_mismatch(self):
        self.record['windows'][0]['input_pcm_sha256']='0'*64;self.save()
        with self.assertRaisesRegex(ValueError,'model input'):self.run_prepare()
    def test_duplicate_sources_rejected_before_network(self):
        self.items*=2;self.save()
        with mock.patch.object(C.subprocess,'run') as call:
            with self.assertRaisesRegex(ValueError,'duplicate'):C.collect(self.sel,self.root/'collect','fixture')
            call.assert_not_called()
    def test_more_than24_rejected(self):
        self.items*=25;self.save()
        with self.assertRaisesRegex(ValueError,'1..24'):self.run_prepare()
    def test_bad_clock(self):
        self.record['windows'][0]['context_end_sample']=8001;self.save()
        with self.assertRaisesRegex(ValueError,'bounded'):self.run_prepare()
    def test_review_ids_cannot_escape(self):
        self.items[0]['review_id']='../escape';self.save()
        with mock.patch.object(C.subprocess,'run') as call:
            with self.assertRaisesRegex(ValueError,'qXX'):C.collect(self.sel,self.root/'collect','fixture')
            call.assert_not_called()
class UnitCropPreparation(unittest.TestCase):
    """候选单元之间漏掉的字仍可在连续原音里，不能靠裁片名称抹去。"""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.audio=self.root/'audio'
        (self.audio/'q01').mkdir(parents=True)
        # 中间区间与两端不同，断言整段PCM可以检测删间隙后再拼接的错误。
        self.wave=np.linspace(-.8,.8,16000,dtype=np.float32)
        self.path=self.audio/'q01/source.wav';sf.write(self.path,self.wave,16000,subtype='FLOAT')
        self.side=self.root/'side.json'
        self.record=dict(source={'source_id':'src_fixture'},
            source_identity=dict(frames=16000,channels=1,sample_rate=16000,
                file_sha256=sha(self.path.read_bytes()),pcm_sha256=sha(self.wave.astype('<f4').tobytes())),
            decision=dict(state='unknown',reasons=['window_0:native_zero_or_negative_duration'],
                usable_for_training=False,usable_for_evaluation=False),
            units=[dict(window_id='w0',native_index=0,text='甲',start_sample=1600,end_sample=3200),
                   dict(window_id='w0',native_index=2,text='丙',start_sample=6400,end_sample=8000)],
            windows=[dict(window_id='w0',context_start_sample=0,context_end_sample=16000,
                asr=dict(raw_text='甲乙丙',normalized_text='甲乙丙'),
                alignment=dict(items=[dict(text='甲'),dict(text='乙'),dict(text='丙')],
                    native_timestamps_ms=[100,200,300,300,400,500],
                    fixed_timestamps_ms=[100,200,300,300,400,500]))])
        self.items=[dict(review_id='q01',local_sidecar=str(self.side),row=dict(source_id='src_fixture',
            audio_path='/fixed/source.wav',audio_recorded_file_sha256=sha(self.path.read_bytes()),
            sidecar_sha256='',decision='unknown'))]
        self.sel=self.root/'selection.json';self.jobs=self.root/'requests.json'
        self.requests=[dict(review_id='q01',edge='first',count=2)];self.save()

    def save(self):
        self.side.write_text(json.dumps(self.record,ensure_ascii=False))
        self.items[0]['row']['sidecar_sha256']=sha(self.side.read_bytes())
        self.sel.write_text(json.dumps(self.items,ensure_ascii=False))
        self.jobs.write_text(json.dumps(self.requests))

    def run_crop(self,name='crop'):
        result=P.prepare_unit_crops(self.sel,self.audio,self.root/name,self.jobs)
        manifest=json.loads(Path(result['manifest']).read_bytes())
        self.assertEqual(result['manifest_sha256'],sha(Path(result['manifest']).read_bytes()))
        return manifest

    def test_continuous_span_keeps_audio_of_excluded_zero_duration_native_unit(self):
        before=self.side.read_bytes();m=self.run_crop();c=m['cases'][0]
        self.assertEqual(c['source_input_sample_bounds'],[1600,8000])
        self.assertEqual([u['native_index'] for u in c['selected_units']],[0,2])
        self.assertEqual(c['reference_text'],'甲 丙')
        actual,sr=sf.read(c['wavpath'],dtype='float32')
        self.assertEqual(sr,16000);np.testing.assert_array_equal(actual,self.wave[1600:8000])
        np.testing.assert_array_equal(actual[1600:4800],self.wave[3200:6400])
        self.assertEqual(c['pcm_float32le_sha256'],sha(self.wave[1600:8000].astype('<f4').tobytes()))
        self.assertEqual(c['file_sha256'],sha(Path(c['wavpath']).read_bytes()))
        self.assertTrue(c['diagnostic_candidate_crop']);self.assertFalse(c['human_gold'])
        self.assertFalse(m['human_gold']);self.assertEqual(self.side.read_bytes(),before)
        self.assertTrue(c['continuous_source_audio_preserved'])
        self.assertFalse(c['complete_word_boundary_verified'])
        self.assertFalse(c['usable_for_training']);self.assertFalse(c['usable_for_evaluation'])
        self.assertFalse(m['usable_for_training']);self.assertFalse(m['usable_for_evaluation'])
        self.assertEqual(c['source_decision_detail']['state'],'unknown')
        self.assertEqual(c['selected_unit_native_indices'],[0,2])
        self.assertEqual(c['selection_sha256'],sha(self.sel.read_bytes()))
        self.assertEqual(c['crop_requests_sha256'],sha(self.jobs.read_bytes()))
        native=c['source_windows_alignment'][0]['alignment']
        self.assertEqual(native['items'][1]['text'],'乙')
        self.assertEqual(native['native_timestamps_ms'][2:4],[300,300])

    def test_first_and_last_select_source_samples_without_padding_or_reordering(self):
        self.requests=[dict(review_id='q01',edge='first',count=1),dict(review_id='q01',edge='last',count=1)]
        self.save();m=self.run_crop()
        self.assertEqual(m['source_ids'],['src_fixture'])
        self.assertEqual([c['source_input_sample_bounds'] for c in m['cases']],[[1600,3200],[6400,8000]])
        for c,(a,b) in zip(m['cases'],[(1600,3200),(6400,8000)]):
            self.assertTrue(c['artificial_context_cut_left']);self.assertTrue(c['artificial_context_cut_right'])
            actual,sr=sf.read(c['wavpath'],dtype='float32');np.testing.assert_array_equal(actual,self.wave[a:b])

    def test_non16k_source_bounds_stay_on_original_clock(self):
        from scipy.signal import resample_poly
        self.wave=self.wave[::2].copy();sf.write(self.path,self.wave,8000,subtype='FLOAT')
        self.record['source_identity'].update(frames=8000,sample_rate=8000,
            file_sha256=sha(self.path.read_bytes()),pcm_sha256=sha(self.wave.astype('<f4').tobytes()))
        self.items[0]['row']['audio_recorded_file_sha256']=sha(self.path.read_bytes())
        for u in self.record['units']:
            u['start_sample']//=2;u['end_sample']//=2
        self.record['windows'][0]['context_end_sample']=8000
        self.save();c=self.run_crop()['cases'][0]
        self.assertEqual(c['source_sample_rate'],8000)
        self.assertEqual(c['source_input_sample_bounds'],[800,4000])
        expected=resample_poly(self.wave[800:4000],2,1).astype('<f4')
        actual,sr=sf.read(c['wavpath'],dtype='float32')
        self.assertEqual(sr,16000);self.assertEqual(c['frames'],6400)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(c['pcm_float32le_sha256'],sha(expected.tobytes()))

    def test_source_encoded_PCM_and_sidecar_identity_must_all_agree(self):
        import copy
        original=copy.deepcopy(self.record);items=copy.deepcopy(self.items)
        for kind in ('encoded','PCM','source_id','sidecar_SHA'):
            with self.subTest(kind=kind):
                self.record=copy.deepcopy(original);self.items=copy.deepcopy(items)
                if kind=='encoded':self.items[0]['row']['audio_recorded_file_sha256']='0'*64
                if kind=='PCM':self.record['source_identity']['pcm_sha256']='0'*64
                if kind=='source_id':self.record['source']['source_id']='src_other'
                self.save()
                if kind=='sidecar_SHA':
                    self.items[0]['row']['sidecar_sha256']='0'*64;self.sel.write_text(json.dumps(self.items))
                with self.assertRaises(ValueError):self.run_crop('crop-'+kind)

    def test_counts_and_request_budget_are_strict_and_foreign_source_rejected(self):
        for kind in ('bool_count','zero_count','seven_count','too_many_units','too_many_jobs','foreign'):
            with self.subTest(kind=kind):
                self.requests=[dict(review_id='q01',edge='first',count=2)]
                if kind=='bool_count':self.requests[0]['count']=True
                if kind=='zero_count':self.requests[0]['count']=0
                if kind=='seven_count':self.requests[0]['count']=7
                if kind=='too_many_units':self.requests[0]['count']=3
                if kind=='too_many_jobs':self.requests*=49
                if kind=='foreign':self.requests[0]['review_id']='q02'
                self.save()
                with self.assertRaises(ValueError):self.run_crop('crop-'+kind)

    def test_selection_cannot_alias_sources_or_escape_review_directory(self):
        import copy
        original=copy.deepcopy(self.items)
        for kind in ('duplicate_source','unsafe_review_id','too_many_sources'):
            with self.subTest(kind=kind):
                self.items=copy.deepcopy(original);self.requests=[dict(review_id='q01',edge='first',count=2)]
                if kind=='duplicate_source':
                    second=copy.deepcopy(self.items[0]);second['review_id']='q02';self.items.append(second)
                if kind=='unsafe_review_id':
                    self.items[0]['review_id']='../escape';self.requests[0]['review_id']='../escape'
                if kind=='too_many_sources':
                    self.items=[dict(copy.deepcopy(original[0]),review_id='q%02d'%(i+1)) for i in range(25)]
                self.save()
                with self.assertRaises(ValueError):self.run_crop('crop-'+kind)

    def test_invalid_unit_span_refused_without_clipping(self):
        import copy
        original=copy.deepcopy(self.record['units'])
        for kind in ('negative','outside','reversed'):
            with self.subTest(kind=kind):
                self.record['units']=copy.deepcopy(original)
                if kind=='negative':self.record['units'][0]['start_sample']=-1
                if kind=='outside':self.record['units'][-1]['end_sample']=16001
                if kind=='reversed':self.record['units'][-1]['end_sample']=1000
                self.save()
                with self.assertRaises(ValueError):self.run_crop('crop-'+kind)

    def test_existing_output_and_original_files_never_overwritten(self):
        m=self.run_crop();out=self.root/'crop'
        before={str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}
        original=self.path.read_bytes()
        with self.assertRaises(ValueError):self.run_crop()
        self.assertEqual(original,self.path.read_bytes())
        self.assertEqual(before,{str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()})

    def test_crop_ledger_hashes_describe_the_bytes_actually_parsed(self):
        # 解码期间调用者文件发生变化时，不能给旧请求结果贴上新清单摘要。
        selected_pin=sha(self.sel.read_bytes());requested_pin=sha(self.jobs.read_bytes())
        real_read=P.sf.read
        def change_ledgers(*args,**kwargs):
            result=real_read(*args,**kwargs)
            self.sel.write_text(json.dumps(self.items,ensure_ascii=False,indent=3))
            self.jobs.write_text(json.dumps([dict(review_id='q01',edge='last',count=1)]))
            return result
        with mock.patch.object(P.sf,'read',side_effect=change_ledgers):
            c=self.run_crop()['cases'][0]
        self.assertEqual(c['source_input_sample_bounds'],[1600,8000])
        self.assertEqual(c['selection_sha256'],selected_pin)
        self.assertEqual(c['crop_requests_sha256'],requested_pin)

if __name__=='__main__':unittest.main()
