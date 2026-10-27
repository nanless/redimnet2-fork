"""仅使用合成文件复核速度计账和差异保留，不执行模型或GPU。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_speed_comparison as A
from contracts import canonical_bytes, digest, job_key, model_provenance, sha_file, stat_identity
from sidecar import SidecarStore
from work_queue import WorkQueue
from test_quality_schema import fixture
from schema import build_record


def pair():
    args = fixture(); a = build_record(*args); b = deepcopy(a)
    for r in (a,b):
        r['windows'][0].update(model_input_frames=32000, resampling_policy='synthetic',
                               preprocessing_status='completed', asr_cache_hit=False, aligner_cache_hit=False)
    return a,b


def contract():
    x = fixture()[2]
    return {'schema':'redimnet2-production-contract/v2','version':'test-v1','purpose':'pilot_validation',
            'model_API_calls_allowed':0,'release_training_labels':False,
            'runtime':{'max_batch':1,'execution_policy':'single_window_independent','dtype':'float32','attention':'eager'},
            'models':x['provenance']['models'],'source_files':{},'alignment':{'mode':'fixture'},'decoder':{},
            'quality_policy':{'synthetic':True},'engine_source_files':{}}


def track(records, seconds=10, cold=True):
    c=contract(); rec={r['source']['source_id']:r for r in records}
    return {'contract':c,'contract_sha256':'a'*64,'spec_sha256':'b'*64,'records':rec,
            'decision_counts':dict(__import__('collections').Counter(r['decision']['state'] for r in records)),
            'performance':{'all_sources_recomputed_cold':cold,'parent_wall_seconds':seconds,
                'actual_compute_execution_verified':True,'totals':{'asr_rows':1}}}


class SpeedComparisonTests(unittest.TestCase):
    def test_ASR_and_native_FIX_differences_remain_distinct(self):
        a,b=pair();b['windows'][0]['asr']['raw_text']='另一个';b['windows'][0]['asr']['native']['generated_token_ids']=[9]
        b['windows'][0]['alignment']['native_timestamps_ms'][0]=120
        r=A.compare_record(a,b);w=r['windows'][0]
        self.assertFalse(w['ASR']['raw_text_equal']);self.assertFalse(w['ASR']['generated_token_ids_equal'])
        self.assertFalse(w['alignment']['native_timestamps_ms_equal']);self.assertTrue(w['alignment']['fixed_timestamps_ms_equal'])
        self.assertEqual(w['alignment']['native_timestamps_ms_max_absolute_delta_ms'],20)
        self.assertEqual(w['alignment']['fixed_timestamps_ms_max_absolute_delta_ms'],0)

    def test_changed_target_does_not_pair_unrelated_native_items(self):
        a,b=pair();b['windows'][0]['alignment']['target_text']='别的'
        w=A.compare_record(a,b)['windows'][0]
        self.assertEqual(w['alignment']['endpoint_comparison_status'],'unmeasured_target_or_geometry_differs')
        self.assertNotIn('fixed_timestamps_ms_max_absolute_delta_ms',w['alignment'])

    def test_missing_windows_failed_sources_and_input_drift_are_counted(self):
        a,b=pair();b['windows']=[];b['decision']['state']='failed';b['source_identity']['decode_status']='decode_failed'
        r=A.analyze(track([a]),track([b],seconds=5))
        self.assertEqual(r['source_count'],1);self.assertEqual(r['counts']['missing_windows'],1)
        self.assertEqual(r['candidate_decision_counts'],{'failed':1});self.assertFalse(r['cold_speed_ratio_valid'])
        a,b=pair();b['windows'][0]['input_pcm_sha256']='f'*64
        r=A.analyze(track([a]),track([b],seconds=5));self.assertEqual(r['counts']['input_mismatch_windows'],1)
        self.assertIsNone(r['cold_end_to_end_speed_ratio'])

    def test_reused_outputs_never_become_cold_speed(self):
        a,b=pair();r=A.analyze(track([a]),track([b],seconds=2,cold=False))
        self.assertIsNone(r['cold_end_to_end_speed_ratio']);self.assertFalse(r['human_text_or_boundary_accuracy_certified'])
        self.assertFalse(r['training_or_evaluation_eligibility_released'])
        r=A.analyze(track([a]),track([b],seconds=5));self.assertEqual(r['cold_end_to_end_speed_ratio'],2)

    def test_recovery_ignores_only_cache_flags_not_model_observations(self):
        a,b=pair();b['windows'][0]['asr_cache_hit']=True
        r=A.analyze(track([a]),track([b],cold=False),mode='recovery')
        self.assertTrue(r['strict_recovery_native_results_equal'])
        b['windows'][0]['asr']['native']['generated_token_ids']=[8]
        r=A.analyze(track([a]),track([b],cold=False),mode='recovery')
        self.assertFalse(r['strict_recovery_native_results_equal'])

    def test_recovery_retains_complete_source_and_producer_identity(self):
        a,b=pair();b['source_identity']['decoder']={'different':'producer'}
        r=A.analyze(track([a]),track([b]),mode='recovery')
        self.assertFalse(r['strict_recovery_native_results_equal'])
        a,b=pair();b['provenance']['code_sha256']='9'*64
        r=A.analyze(track([a]),track([b]),mode='recovery')
        self.assertFalse(r['strict_recovery_native_results_equal'])

    def test_ASR_compute_only_contract_change_and_recovery_same_contract(self):
        a,b=contract(),contract();b['runtime']['asr_compute']={'dtype':'bfloat16','attention':'sdpa'}
        A.compare_contracts(a,b,'speed')
        with self.assertRaises(RuntimeError):A.compare_contracts(a,b,'recovery')
        for field in ('decoder','models','source_files','alignment','quality_policy','engine_source_files'):
            x=deepcopy(b);x[field]={'different':True}
            with self.subTest(field=field),self.assertRaises(RuntimeError):A.compare_contracts(a,x,'speed')
        x=deepcopy(b);x['runtime']['max_batch']=4
        with self.assertRaises(RuntimeError):A.compare_contracts(a,x,'speed')

    def test_source_union_and_source_identity_not_silently_filtered(self):
        a,b=pair();c=deepcopy(b);c['source']['source_id']='src_'+'8'*64
        with self.assertRaises(RuntimeError):A.analyze(track([a]),track([c]))
        with self.assertRaises(RuntimeError):A.compare_record(a,c)
        b['source_identity']['pcm_sha256']='4'*64
        r=A.analyze(track([a]),track([b]));self.assertFalse(r['cold_speed_ratio_valid'])
        self.assertEqual(r['counts']['source_identity_changed'],1)

    def compute_metrics(self, profile=None):
        profile = profile or {'dtype':'float32','attention':'sdpa'}
        def loaded(x):
            return {'requested':x,'floating_parameter_elements_by_dtype':{x['dtype']:100},
                    'attention_configurations':[{'module':'module.config','attention_config':x['attention']}]}
        return {'asr_rows':2,'first_call_trace_completed':True,
                'loaded_compute':{'asr':loaded(profile),'aligner':loaded({'dtype':'float32','attention':'eager'})},
                'first_call_compute_observation':{'attention_forwards':{'audio':1,'text':2},
                    'functional_sdpa_calls':{'audio':1,'text':2},
                    'attention_module_configurations':{'audio':'sdpa','text':'sdpa'},'attention_configurations_stable':True,
                    'functional_sdpa_qkv_dtypes':{'audio':[profile['dtype']]*3,'text':[profile['dtype']]*3},
                    'functional_sdpa_qkv_dtypes_stable':True}}

    def test_explicit_dispatch_profile_must_have_new_calls_actual_parameters_and_trace(self):
        c=contract();c['runtime']['asr_compute']={'dtype':'float32','attention':'sdpa'}
        m=self.compute_metrics();self.assertTrue(A.compute_execution(m,c)['verified'])
        for field in ('loaded_compute','first_call_compute_observation','first_call_trace_completed'):
            q=deepcopy(m);q.pop(field)
            self.assertFalse(A.compute_execution(q,c)['verified'])
        q=deepcopy(m);q['asr_rows']=0
        self.assertFalse(A.compute_execution(q,c)['verified'])
        q=deepcopy(m);q['loaded_compute']['asr']['floating_parameter_elements_by_dtype']={'bfloat16':100}
        with self.assertRaises(RuntimeError):A.compute_execution(q,c)
        q=deepcopy(m);q['first_call_compute_observation']['functional_sdpa_calls']['text']=1
        with self.assertRaises(RuntimeError):A.compute_execution(q,c)
        q=deepcopy(m);q['first_call_compute_observation']['functional_sdpa_qkv_dtypes']['text']=['bfloat16']*3
        with self.assertRaises(RuntimeError):A.compute_execution(q,c)

    def test_legacy_mixed_actual_dispatch_is_observed_without_rewriting_it(self):
        c=contract();m=self.compute_metrics({'dtype':'float32','attention':'eager'})
        obs=m['first_call_compute_observation'];obs['attention_module_configurations']['text']='eager';obs['functional_sdpa_calls']['text']=0
        before=deepcopy((c,m));r=A.compute_execution(m,c)
        self.assertTrue(r['verified']);self.assertFalse(r['profile_declared']);self.assertTrue(r['observed_nonuniform_attention_dispatch'])
        self.assertEqual((c,m),before)
        obs['functional_sdpa_calls']['text']=1
        with self.assertRaises(RuntimeError):A.compute_execution(m,c)

    def test_legacy_without_module_config_and_unstable_config_not_verified(self):
        c=contract();m=self.compute_metrics({'dtype':'float32','attention':'eager'})
        m['first_call_compute_observation'].pop('attention_module_configurations')
        self.assertFalse(A.compute_execution(m,c)['verified'])
        m=self.compute_metrics();m['first_call_compute_observation']['attention_configurations_stable']=False
        with self.assertRaises(RuntimeError):A.compute_execution(m,c)

    def test_no_comparable_endpoints_is_unmeasured_not_zero_error(self):
        a,b=pair();b['windows'][0]['alignment']['target_text']='different'
        r=A.analyze(track([a]),track([b]))
        self.assertIsNone(r['maximum_same_target_endpoint_delta_ms']['native_timestamps_ms'])
        self.assertFalse(r['formal_admission']);self.assertFalse(r['changed_rows_are_accuracy_errors'])

    def test_CPU_fixture_never_yields_real_cold_speed(self):
        a,b=pair();ta,tb=track([a]),track([b],seconds=5)
        ta['contract']['purpose']=tb['contract']['purpose']='cpu_fixture'
        self.assertIsNone(A.analyze(ta,tb)['cold_end_to_end_speed_ratio'])

    def test_report_path_refusal_is_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();out=root/'report';out.write_text('keep');link=root/'link';link.symlink_to(root/'missing')
            for p in (out,link):
                with self.subTest(path=p),patch.object(A,'load_track') as load,self.assertRaises(RuntimeError):
                    A.main(['--baseline-spec','unused','--baseline-spec-sha','a'*64,'--candidate-spec','unused',
                            '--candidate-spec-sha','b'*64,'--output',str(p)])
                load.assert_not_called()
            self.assertEqual(out.read_text(),'keep')

    def test_package_import_uses_local_maintained_modules(self):
        import subprocess
        r=subprocess.run([sys.executable,'-B','-c','from tools.open_annotation.audit_speed_comparison import compare_record'],
                         cwd=Path(__file__).resolve().parents[3],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)


class FullReadbackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        c=contract();c['purpose']='cpu_fixture';cp=self.root/'contract.json';cp.write_bytes(canonical_bytes(c));self.csha=sha_file(cp)
        args=list(fixture());wav=self.root/'a.wav'
        with wave.open(str(wav),'wb') as f:f.setnchannels(2);f.setsampwidth(2);f.setframerate(44100);f.writeframes(bytes(88200*4))
        args[0]['audio_path']=str(wav);args[1].update(bytes=wav.stat().st_size,file_sha256=sha_file(wav),stat_before=stat_identity(wav),stat_after=stat_identity(wav))
        args[2]={'sha256':self.csha,'version':c['version'],'api_calls':0,'provenance':{'models':model_provenance(c),'runtime':c['runtime'],'code_sha256':'6'*64}}
        args[4][0].update(model_input_frames=32000,resampling_policy='synthetic',preprocessing_status='completed',asr_cache_hit=False,aligner_cache_hit=False)
        shard=self.root/'chunk.jsonl';shard.write_bytes(canonical_bytes(args[0]));index=self.root/'index.json';index.write_bytes(canonical_bytes({'schema':'redimnet2-shard-index/v1','source_count':1,'shards':[{'id':'part-000','path':shard.name,'count':1,'sha256':sha_file(shard)}]}));isha=sha_file(index)
        qroot=self.root/'queue';q=WorkQueue(index,isha,qroot,self.csha);store=SidecarStore(self.root/'labels',c,self.csha)
        with q.claim_next('worker-0') as claim:
            args[6].update(job_key=job_key(args[0],self.csha),attempt_id=claim.attempt_id,worker_id='worker-0')
            record=build_record(*args);receipt=store.publish(args[0],record,claim);claim.finish([receipt])
        self.path=store.path(args[0]);res=self.root/'resources';worker=res/'worker-0';worker.mkdir(parents=True)
        summary={'worker_id':'worker-0','contract_sha256':self.csha,'index_sha256':isha,'device_uuid':'synthetic-no-GPU',
                 'assigned_source_rows':1,'sidecars_committed':1,'sidecar_cache_hits':0,'model_windows':1,'decoded_audio_seconds':2.,
                 'stage_cache_hits':{'asr':0,'aligner':0},'elapsed_seconds':3.,'model_API_calls':0,
                 'chunk_receipt_refs':[{'id':'part-000','sha256':sha_file(qroot/'completed/part-000.json'),'source_count':1}],
                 'backend_metrics':{'model_API_calls':0,'asr_rows':1,'aligner_rows':1,'asr_batches':1,'aligner_batches':1,
                    'asr_seconds':2.,'aligner_seconds':.2,'model_load_seconds':.1,'asr_call_rows':{'1':1},'aligner_call_rows':{'1':1}}}
        (worker/'worker_summary.json').write_bytes(canonical_bytes(summary));ready={'summary_sha256':sha_file(worker/'worker_summary.json'),
            'contract_sha256':self.csha,'index_sha256':isha,'bundle_sha256':'6'*64,'nonce':'synthetic'}
        (worker/'terminal_ready.json').write_bytes(canonical_bytes(ready));ack={'ready_sha256':sha_file(worker/'terminal_ready.json'),'nonce':'synthetic'}
        (worker/'terminal_ack.json').write_bytes(canonical_bytes(ack))
        parent={'resource_returned':True,'full_sidecar_readback':True,'worker_count':1,'contract_sha256':self.csha,'index_sha256':isha,
                'bundle_sha256':'6'*64,'model_API_calls':0,'parent_elapsed_seconds':4.,'gpu_lease_wall_seconds':3.5,
                'workers':[{'worker_id':'worker-0','uuid':'synthetic-no-GPU','terminal':{'summary':summary,'summary_sha256':sha_file(worker/'worker_summary.json'),
                            'ready_sha256':sha_file(worker/'terminal_ready.json')}}]}
        (res/'parent_result.json').write_bytes(canonical_bytes(parent))
        self.spec={'schema':A.SPEC_SCHEMA,'contract':{'path':str(cp),'sha256':self.csha},'index':{'path':str(index),'sha256':isha},
                   'queue_root':str(qroot),'output_root':str(self.root/'labels'),'resources_root':str(res)}
        self.specpath=self.root/'spec.json';self.specpath.write_bytes(canonical_bytes(self.spec));self.specsha=sha_file(self.specpath)

    def tearDown(self):self.temp.cleanup()

    def test_complete_publications_are_read_without_touching_queue(self):
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        t=A.load_track(self.specpath,self.specsha)
        self.assertEqual(len(t['records']),1);self.assertEqual(t['performance']['totals']['sidecars_committed'],1)
        self.assertEqual(before,{p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_full_CPU_fixture_CLI_writes_private_comparison_but_never_GPU_speed_admission(self):
        out=self.root/'comparison.json'
        rc=A.main(['--baseline-spec',str(self.specpath),'--baseline-spec-sha',self.specsha,
                   '--candidate-spec',str(self.specpath),'--candidate-spec-sha',self.specsha,'--output',str(out)])
        self.assertEqual(rc,0);report=json.loads(out.read_text())
        self.assertEqual(report['source_count'],1);self.assertTrue(report['full_sidecar_commit_readback_verified'])
        self.assertFalse(report['formal_admission']);self.assertIsNone(report['cold_end_to_end_speed_ratio'])
        self.assertFalse(report['baseline_performance']['historical_resource_digest_chain_verified'])

    def test_modified_sidecar_and_wrong_spec_SHA_are_rejected(self):
        with self.assertRaises(RuntimeError):A.load_track(self.specpath,'a'*64)
        self.path.write_bytes(self.path.read_bytes()+b' ')
        with self.assertRaises(RuntimeError):A.load_track(self.specpath,self.specsha)

    def test_running_incomplete_track_is_not_success(self):
        p=self.root/'queue/broker.json';x=json.loads(p.read_text());x['completed_chunks']=0;p.write_bytes(canonical_bytes(x))
        with self.assertRaises((RuntimeError,ValueError)):A.load_track(self.specpath,self.specsha)

    def test_input_ancestor_symlink_refused(self):
        link=self.root/'alias';link.symlink_to(self.root/'labels',target_is_directory=True)
        with self.assertRaises(RuntimeError):A.bound(link/'test-v1/identity.json')


if __name__=='__main__':unittest.main()
