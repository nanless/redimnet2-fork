"""只用合成PCM验证两次读取、完整身份和有限CPU时延入口。"""
import importlib.util
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import sys
import unittest
from unittest import mock
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PATH=Path(__file__).resolve().parents[1]/'bounded_cpu_audio_timing.py'
spec=importlib.util.spec_from_file_location('bounded_cpu_audio_timing',PATH)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
sha=lambda raw:hashlib.sha256(raw).hexdigest()

class BoundedTiming(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.audio=self.root/'fixture.wav';self.data=np.column_stack([np.sin(np.arange(16000)*.03),np.sin(np.arange(16000)*.07)]).astype(np.float32)
        sf.write(self.audio,self.data,16000,subtype='FLOAT')
        self.source=dict(source_id='fixture-one',audio_path=str(self.audio),bytes=self.audio.stat().st_size,frames=len(self.data),channels=2,sample_rate=16000,file_sha256=sha(self.audio.read_bytes()),pcm_sha256=sha(self.data.astype('<f4').tobytes()),windows=[])
        for index,(a,b)in enumerate(((0,12000),(8000,16000))):
            pcm=self.data[a:b].mean(axis=1,dtype=np.float32)
            self.source['windows'].append(dict(window_id=str(index),context_start_sample=a,context_end_sample=b,input_pcm_sha256=sha(pcm.astype('<f4').tobytes())))
        self.value=dict(schema='bounded-cpu-audio-timing-input/v1',sources=[self.source])
    def run_measure(self):return M.measure_sources(self.value,socket.gethostname())
    def test_exact_identity_two_opens(self):
        original=M.os.open
        with mock.patch.object(M.os,'open',wraps=original)as opened:r=self.run_measure()
        self.assertTrue(r['passed']);self.assertEqual(r['physical_source_opens'],2);self.assertEqual(opened.call_count,2)
        self.assertEqual(r['sources'][0]['file_sha256'],self.source['file_sha256']);self.assertEqual(len(r['sources'][0]['windows']),2)
        self.assertFalse(r['GPU_context_created']);self.assertFalse(r['actual_cloud_stage_profile']);self.assertFalse(r['cold_cache_claim'])
        for key in ('encoded_read','encoded_hash','memory_decode_and_pcm_hash','second_file_decode','second_pcm_hash'):
            self.assertGreaterEqual(r['sources'][0][key]['wall_seconds'],0)
    def test_bad_encoded_identity_stops_before_decode(self):
        self.source['file_sha256']='0'*64;r=self.run_measure();self.assertFalse(r['passed']);self.assertEqual(r['physical_source_opens'],1);self.assertIn('encoded original identity',r['sources'][0]['traceback'])
    def test_bad_pcm_identity_preserves_failure(self):
        self.source['pcm_sha256']='0'*64;r=self.run_measure();self.assertFalse(r['passed']);self.assertEqual(r['physical_source_opens'],1);self.assertIn('first native PCM identity',r['sources'][0]['traceback'])
    def test_bad_context_pcm(self):
        self.source['windows'][0]['input_pcm_sha256']='0'*64;r=self.run_measure();self.assertFalse(r['passed']);self.assertEqual(r['physical_source_opens'],2);self.assertIn('context model PCM',r['sources'][0]['traceback'])
    def test_changed_source_between_passes(self):
        original=M.os.open;count=0
        def opening(path,*args,**kwargs):
            nonlocal count
            count+=1
            if count==2:
                with self.audio.open('ab')as stream:stream.write(b'changed')
            return original(path,*args,**kwargs)
        with mock.patch.object(M.os,'open',side_effect=opening):r=self.run_measure()
        self.assertFalse(r['passed']);self.assertIn('source changed before second read',r['sources'][0]['traceback'])
    def test_invalid_source_count(self):
        self.value['sources']=[dict(self.source,source_id=str(i))for i in range(25)]
        with self.assertRaisesRegex(ValueError,'1..24'):self.run_measure()
    def test_duplicate_and_illegal_context(self):
        self.value['sources']=[self.source,self.source]
        with self.assertRaisesRegex(ValueError,'duplicate'):self.run_measure()
        self.value['sources']=[self.source];self.source['windows'][0]['context_end_sample']=16001
        with self.assertRaisesRegex(ValueError,'context'):self.run_measure()
    def test_host_must_match(self):
        with self.assertRaisesRegex(ValueError,'host differs'):M.measure_sources(self.value,'not-the-current-node')
    def test_non16k_source_preserves_native_and_model_clocks(self):
        data=self.data[::2];sf.write(self.audio,data,8000,subtype='FLOAT')
        self.source.update(bytes=self.audio.stat().st_size,frames=len(data),sample_rate=8000,file_sha256=sha(self.audio.read_bytes()),pcm_sha256=sha(data.astype('<f4').tobytes()))
        mono=resample_poly(data.mean(axis=1,dtype=np.float32),2,1).astype('<f4')
        self.source['windows']=[dict(window_id='resample',context_start_sample=0,context_end_sample=len(data),input_pcm_sha256=sha(mono.tobytes()))]
        r=self.run_measure();self.assertTrue(r['passed']);self.assertEqual(r['sources'][0]['windows'][0]['model_frames'],16000)
    def test_repository_cpu_helpers_are_not_heavy_model_packages(self):
        # 用真正的维护模块复现整套 discover 的导入顺序；不删模块躲守卫。
        import vllm_backend
        import vllm_live_probe
        self.assertIs(sys.modules['vllm_backend'],vllm_backend)
        self.assertIs(sys.modules['vllm_live_probe'],vllm_live_probe)
        self.assertTrue(self.run_measure()['passed'])
        self.assertIs(sys.modules['vllm_backend'],vllm_backend)
    def test_real_model_package_names_and_submodules_still_refused_before_audio_open(self):
        for name in ('torch','torch.nn','qwen_asr','qwen_asr.inference','vllm','vllm.engine'):
            marker=object()
            with self.subTest(name=name),mock.patch.dict(sys.modules,{name:marker}),\
                 mock.patch.object(M.os,'open',side_effect=AssertionError('must reject before original audio open')):
                with self.assertRaisesRegex(ValueError,'heavy model module already imported'):
                    self.run_measure()
                self.assertIs(sys.modules[name],marker)

if __name__=='__main__':unittest.main()
