"""只用合成收据核新原生批次1/4/8串行试验；生命周期复用旧已验SerialExecutor；不启动GPU或接触现有生产。"""
from pathlib import Path
import copy,json,os,sys,tempfile,threading,unittest,signal,time
from unittest.mock import patch
from types import SimpleNamespace
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_native_batch_pilot as D
import prepare_native_batch_validation as P
from contracts import canonical_bytes,sha_file
from source_registry import source_id
from test_maintenance_entries import external_contract

class PlanFixture(unittest.TestCase):
    def setUp(self):
        # 本夹具只测调度/失败/恢复顺序；完整原生收据由独立audit文件夹具实测。
        readback=patch.object(D,'save_native_readback',return_value='d'*64)
        readback.start();self.addCleanup(readback.stop)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.python=self.root/'python';self.python.write_text('CPU fixture; not executable')
        self.rows=[{'source_id':source_id(ds,f's/{i}.wav'),'dataset':ds,'relative_path':f's/{i}.wav','roles':['train_ptn'],'sidecar_relpath':f'{ds}/{i}.annotation.json'}for i in range(3000)for ds in [('voxceleb2','voxblink2','cnceleb2')[i%3]]]
        self.refs={};self.prepared={}
        base=self.root/'parent-input/base.json';base.parent.mkdir();base.write_bytes(canonical_bytes(external_contract()))
        for name,count in [('screen',30),('main',3000)]:
            folder=self.root/(name+'-inputs');folder.mkdir();shard=folder/'s.jsonl';shard.write_bytes(b''.join(canonical_bytes(x)for x in self.rows[:count]))
            index=folder/'index.json';index.write_bytes(canonical_bytes({'schema':'redimnet2-shard-index/v1','source_count':count,'shards':[{'id':'s','count':count,'path':shard.name,'sha256':sha_file(shard)}]}))
            spec={'schema':P.SCHEMA,'source_count':count,'base_contract':{'path':str(base),'sha256':sha_file(base)},'index':{'path':str(index),'sha256':sha_file(index)},'target_uuids':'auto2','python':str(self.python),'device_lock_root':str(self.root/'global-locks'),'protected_roots':[str(self.root/'running-annotation'),str(self.root/'PTN')],'max_runtime_seconds':1800 if count==30 else 7200}
            specpath=self.root/(name+'-spec.json');specpath.write_bytes(canonical_bytes(spec))
            with patch.object(P.freezer,'freeze',side_effect=self.freeze):result=P.prepare(specpath,sha_file(specpath),self.root/(name+'-prepared'))
            path=self.root/(name+'-prepared')/'prepared.json';self.refs[name]={'path':str(path),'sha256':sha_file(path)};self.prepared[name]=result
        self.plan={'schema':D.SCHEMA,'purpose':'cpu_fixture','model_API_calls_allowed':0,'production_switch_allowed':False,'screen_prepared':self.refs['screen'],'main_prepared':self.refs['main'],'protected_roots':[str(self.root/'running-annotation'),str(self.root/'PTN')],'wall_budget_seconds':21000}
        self.planpath=self.root/'plan.json';self.out=self.root/'driver-evidence';self.save_plan()
    def freeze(self,base,pin,output,prefix,asr_compute,asr_batching):
        output.mkdir(parents=True);members={}
        for name in ('gpu_launch.py','report.py','worker.py','audit_native_transformers_batch.py'):
            path=output/name;path.write_text('CPU fixture source; never execute');members[name]=sha_file(path)
        bundle=output/'bundle_files.json';bundle.write_bytes(canonical_bytes(members));c=json.loads(Path(base).read_text());c['runtime']['asr_compute']=asr_compute;c['runtime']['execution_policy']='asr_fixed_cohort_fa_single/v1';c['runtime']['asr_batching']=asr_batching;c['purpose']='pilot_validation';c['version']=prefix
        cp=output/'contracts'/f'{prefix}.json';cp.parent.mkdir();cp.write_bytes(canonical_bytes(c))
        return {'frozen_copy':str(output),'bundle_sha256':sha_file(bundle),'contracts':{'pilot_validation':{'path':str(cp),'sha256':sha_file(cp)}}}
    def save_plan(self):
        self.planpath.write_bytes(canonical_bytes(self.plan));self.ref={'path':str(self.planpath),'sha256':sha_file(self.planpath)}
    def test_precheck_consumes_fixed_30_3000_only_and_creates_nothing(self):
        value=D.run(self.ref,self.out);self.assertTrue(value['checked']);self.assertFalse(self.out.exists());self.assertFalse(value['GPU_executed'])
    def test_prepared_argv_shell_or_extra_flags_cannot_execute(self):
        for mutation in ('shell','extra','wrong-card-count'):
            ref=self.refs['screen'];path=Path(ref['path']);original=path.read_bytes();x=json.loads(original);v=x['variants'][D.PROFILES[0]]
            if mutation=='shell':v['argv']=['/bin/sh','-c','touch /never']
            elif mutation=='extra':v['argv']+=['--unreviewed','true']
            else:x['worker_count']=8
            path.write_bytes(canonical_bytes(x));changed={'path':str(path),'sha256':sha_file(path)}
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):D.load_matrix(changed,30,cpu_fixture=True)
            path.write_bytes(original)
    def test_frozen_source_tamper_rejected_before_execution(self):
        path=self.root/'screen-prepared'/D.PROFILES[0]/'code/gpu_launch.py';path.write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'member'):D.compile_plan(self.ref,self.out)
    def test_other_generation_change_is_refused_before_any_GPU_command(self):
        ref=self.refs['screen'];path=Path(ref['path']);matrix=json.loads(path.read_text())
        variant=matrix['variants'][D.PROFILES[1]]
        cp=Path(variant['freeze']['contracts']['pilot_validation']['path'])
        c=json.loads(cp.read_text());c['runtime']['max_new_tokens']+=1
        cp.write_bytes(canonical_bytes(c));contract_pin={'path':str(cp),'sha256':sha_file(cp)}
        variant['freeze']['contracts']['pilot_validation']=contract_pin
        variant['argv'][variant['argv'].index('--contract-sha')+1]=contract_pin['sha256']
        tp=Path(variant['track']['path']);track=json.loads(tp.read_text());track['contract']=contract_pin
        tp.write_bytes(canonical_bytes(track));variant['track']['sha256']=sha_file(tp)
        path.write_bytes(canonical_bytes(matrix));changed={'path':str(path),'sha256':sha_file(path)}
        with self.assertRaisesRegex(RuntimeError,'only declared ASR batch width'):
            D.load_matrix(changed,30,cpu_fixture=True)
        self.assertFalse(self.out.exists())
    def test_source_subset_and_protected_roots_required(self):
        self.plan['protected_roots']=[str(self.root/'driver-evidence')];self.save_plan()
        with self.assertRaisesRegex(RuntimeError,'protected'):D.compile_plan(self.ref,self.out)
    def test_evidence_cannot_be_written_into_control_source_input_or_locks(self):
        paths=[D.HERE/'must-not-create-native-evidence',
               self.root/'screen-inputs'/'evidence']
        locks=self.root/'global-locks';locks.mkdir();paths.append(locks/'evidence')
        for target in paths:
            with self.subTest(target=target),self.assertRaisesRegex(RuntimeError,'control code'):
                D.compile_plan(self.ref,target)
            self.assertFalse(target.exists())
        # 计划在共同祖先根并不禁止正常兄弟evidence目录。
        self.assertEqual(D.compile_plan(self.ref,self.out)['output'],self.out)
    def test_real_purpose_rejects_custom_executor(self):
        self.plan['purpose']='production';self.save_plan()
        with self.assertRaisesRegex(RuntimeError,'control source differ|custom executors'):D.run(self.ref,self.out,True,_fixture_executor=object())
        self.assertFalse(self.out.exists())
    def test_serial_order_full_audit_before_next_profile_and_stage(self):
        events=[]
        class Executor:
            def command(s,argv,name,code):events.append(('command',name))
        def verified(v):events.append(('return',v['contract']['version']));return {'resource_returned':True}
        def loaded(v):events.append(('readback',v['contract']['version']));return {'records':{str(i):{}for i in range(v['track']['index']==self.prepared['screen']['index'] and 30 or 3000)}}
        with patch.object(D,'verify_parent',side_effect=verified),patch.object(D,'audited_track',side_effect=loaded),patch('audit_native_transformers_batch.analyze',return_value={'all_planned_sources_accounted':True,'cold_speed_ratio_valid':True}):
            result=D.run(self.ref,self.out,True,_fixture_executor=Executor())
        self.assertTrue(result['all_finite_audits_completed']);self.assertFalse(result['GPU_runtime_verified'])
        commands=[e[1]for e in events if e[0]=='command'];self.assertEqual(commands,[s+'-'+n for s in ('screen','main')for n in D.PROFILES])
        self.assertEqual([e[0]for e in events],['command','return','readback']*6)
    def test_failure_or_incomplete_comparison_stops_before_main(self):
        class Executor:
            def __init__(s):s.calls=[]
            def command(s,argv,name,code):s.calls.append(name)
        executor=Executor()
        with patch.object(D,'verify_parent',return_value={'resource_returned':True}),patch.object(D,'audited_track',return_value={'records':{'x':{}}}),patch('audit_native_transformers_batch.analyze',return_value={'all_planned_sources_accounted':True,'cold_speed_ratio_valid':False}):
            with self.assertRaisesRegex(RuntimeError,'cold measurement'):D.run(self.ref,self.out,True,_fixture_executor=executor)
        self.assertEqual(executor.calls,['screen-'+n for n in D.PROFILES]);self.assertTrue((self.out/'failure.json').exists());self.assertFalse((self.out/'result.json').exists())
    def test_parent_failure_never_launches_second_profile(self):
        class Executor:
            def __init__(s):s.calls=[]
            def command(s,argv,name,code):s.calls.append(name);raise RuntimeError('synthetic parent failed')
        executor=Executor()
        with self.assertRaisesRegex(RuntimeError,'parent failed'):D.run(self.ref,self.out,True,_fixture_executor=executor)
        self.assertEqual(len(executor.calls),1);self.assertTrue((self.out/'failure.json').exists())

    def add_recovery(self):
        root=self.root/'recovery-input';root.mkdir();shard=root/'s.jsonl';shard.write_bytes(b''.join(canonical_bytes(r)for r in self.rows[:512]))
        index=root/'index.json';index.write_bytes(canonical_bytes({'schema':'redimnet2-shard-index/v1','source_count':512,'shards':[{'id':'s','path':shard.name,'count':512,'sha256':sha_file(shard)}]}))
        self.plan['recovery_index']={'path':str(index),'sha256':sha_file(index)};self.plan['recovery_profiles']=['native-fp32-sdpa-b8'];self.save_plan()
        v=self.prepared['main']['variants']['native-fp32-sdpa-b8'];contract=json.loads(Path(v['freeze']['contracts']['pilot_validation']['path']).read_text());tree=Path(json.loads(Path(v['track']['path']).read_text())['output_root'])/contract['version']
        for r in self.rows[:512]:
            p=tree/r['sidecar_relpath'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canonical_bytes({'original':r['source_id']}));Path(str(p)+'.commit.json').write_bytes(canonical_bytes({'untouched':True}))
        return tree
    def test_512_recovery_uses_same_labels_contract_new_queue_and_zero_model_calls(self):
        tree=self.add_recovery();calls=[]
        class Executor:
            def command(s,argv,name,code):calls.append((name,argv))
        def loaded(v):
            recovery='recovery512'in v['track']['queue_root']
            return {'records':{str(i):{}for i in range(512 if recovery else 3000)},'performance':{'totals':{'sidecars_committed':0,'sidecar_cache_hits':512,'asr_rows':0,'aligner_rows':0}}}
        def analyze(a,b,mode):return {'all_planned_sources_accounted':True,'cold_speed_ratio_valid':True,'strict_recovery_native_results_equal':True}
        before={str(p):D.file_identity(p)for p in tree.rglob('*')if p.is_file()}
        with patch.object(D,'verify_parent',return_value={'resource_returned':True}),patch.object(D,'audited_track',side_effect=loaded),patch('audit_native_transformers_batch.analyze',side_effect=analyze):
            result=D.run(self.ref,self.out,True,_fixture_executor=Executor())
        self.assertEqual(len(calls),7);self.assertEqual(result['steps'][-1]['sources'],512);self.assertEqual(result['steps'][-1]['new_ASR_calls'],0)
        argv=calls[-1][1];fields=D.argv_fields(argv);self.assertIn('/recovery512/queue',fields['--queue-root']);self.assertEqual(fields['--output-root'],str(tree.parent))
        self.assertEqual(before,{str(p):D.file_identity(p)for p in tree.rglob('*')if p.is_file()})
    def test_recovery_recompute_or_mutated_old_output_is_rejected(self):
        tree=self.add_recovery()
        class Executor:
            def command(s,argv,name,code):
                if name.startswith('recovery-'):next(tree.rglob('*.annotation.json')).write_bytes(b'changed')
        def loaded(v):return {'records':{'x':{}},'performance':{'totals':{'sidecars_committed':0,'sidecar_cache_hits':512,'asr_rows':0,'aligner_rows':0}}}
        with patch.object(D,'verify_parent',return_value={'resource_returned':True}),patch.object(D,'audited_track',side_effect=loaded),patch('audit_native_transformers_batch.analyze',return_value={'all_planned_sources_accounted':True,'cold_speed_ratio_valid':True,'strict_recovery_native_results_equal':True}):
            with self.assertRaisesRegex(RuntimeError,'changed during recovery'):D.run(self.ref,self.out,True,_fixture_executor=Executor())
    def test_cancel_after_child_return_does_not_start_next(self):
        event=threading.Event();calls=[]
        class Executor:
            def command(s,argv,name,code):calls.append(name);event.set()
        with patch.object(D.threading,'Event',return_value=event):
            with self.assertRaisesRegex(RuntimeError,'cancelled after'):D.run(self.ref,self.out,True,_fixture_executor=Executor())
        self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
