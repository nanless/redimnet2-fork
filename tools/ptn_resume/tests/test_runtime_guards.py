"""Runtime control-flow probes only; no models, CUDA, real /proc or launch."""
import contextlib
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from tools.ptn_resume import contracts as C

RUNTIME=Path(__file__).resolve().parents[1]/'runtime'


def load(name,strict):
    spec=importlib.util.spec_from_file_location('runtime_probe_'+name,RUNTIME/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'strict_restore':strict}):spec.loader.exec_module(module)
    return module


class RuntimeGuardTests(unittest.TestCase):
    def strict(self,root):
        s=types.ModuleType('strict_restore');s.C=C;s.need=C.need
        s.PACKAGE=root/'package';s.PACKAGE.mkdir();s.RUN=root/'new-run';s.OLD=root/'original';s.OLD.mkdir()
        s.PREDECESSOR=root/'failed';s.PREDECESSOR.mkdir();s.SOURCE=s.PACKAGE/'source';s.SOURCE.mkdir()
        s.CONFIG={'legacy':{'bundle_root':str(root/'legacy')}};(root/'legacy').mkdir()
        s.configure=lambda *args:None;s.PACKAGE_SHA='a'*64;s.CONTRACT_SHA='c'*64;s.CK_SHA='d'*64
        return s

    def test_entry_and_worker_help_never_import_torch(self):
        code="""import sys,types,runpy,builtins
sys.modules['strict_restore']=types.ModuleType('strict_restore')
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name=='torch' or name.startswith('torch.'):raise RuntimeError('forbidden Torch import during help')
 return original(name,*args,**kwargs)
builtins.__import__=guarded
path=sys.argv[1];sys.argv=[path,'--help'];runpy.run_path(path,run_name='__main__')
"""
        for name in ('entry.py','worker.py'):
            with self.subTest(name=name):
                value=subprocess.run([sys.executable,'-B','-c',code,str(RUNTIME/name)],capture_output=True,text=True,timeout=10,
                    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
                self.assertEqual(value.returncode,0,value.stderr);self.assertIn('usage:',value.stdout)

    def test_preflight_report_cannot_claim_training_run_or_write_old_evidence_trees(self):
        for category in ('new_run_exact','old_child','failed_child','legacy_child','package_child'):
            with self.subTest(category=category),tempfile.TemporaryDirectory()as temp:
                root=Path(temp).resolve();s=self.strict(root);entry=load('entry',s)
                output={'new_run_exact':s.RUN,'old_child':s.OLD/'new-report.json','failed_child':s.PREDECESSOR/'new-report.json',
                    'legacy_child':root/'legacy/new-report.json','package_child':s.PACKAGE/'new-report.json'}[category]
                cpu=types.ModuleType('cpu_check');cpu.run=lambda:{'synthetic_cpu':True}
                argv=['entry.py','--contract','unused','--contract-sha','a'*64,'--package-sha','b'*64,
                      '--preflight-only','--preflight-report',str(output)]
                with patch.dict(sys.modules,{'cpu_check':cpu}),patch.object(sys,'argv',argv),patch.dict(os.environ,{},clear=False),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError):entry.main()
                self.assertFalse(output.exists())

    def test_gpu_parser_requires_eight_distinct_idle_A800_and_no_compute_PID(self):
        with tempfile.TemporaryDirectory()as temp:
            entry=load('entry',self.strict(Path(temp).resolve()))
            rows=['%d,GPU-00000000-0000-0000-0000-%012d,NVIDIA A800,555.42.02,81920,2,81918'%(i,i+1)for i in range(8)]
            def result(gpus,processes=''):
                return [types.SimpleNamespace(returncode=0,stderr='',stdout='\n'.join(gpus)+'\n'),
                        types.SimpleNamespace(returncode=0,stderr='',stdout=processes)]
            with patch.object(entry.subprocess,'run',side_effect=result(rows)):
                self.assertEqual(len(entry.gpu_snapshot()['devices']),8)
            for gpus,processes in ((rows,'GPU-unknown,123,unknown,100\n'),(rows[:7],''),
                 ([row.replace('NVIDIA A800','NVIDIA other')for row in rows],''),(rows[:7]+[rows[6]],'')):
                with self.subTest(gpus=len(gpus),processes=processes),patch.object(entry.subprocess,'run',side_effect=result(gpus,processes)):
                    with self.assertRaises(ValueError):entry.gpu_snapshot()

    def test_formal_mounts_require_gpfs_root_rw_and_shared_ro(self):
        with tempfile.TemporaryDirectory()as temp:
            entry=load('entry',self.strict(Path(temp).resolve()));actual_path=Path
            alias=types.SimpleNamespace(exists=lambda:True,is_symlink=lambda:False,is_dir=lambda:True,samefile=lambda other:True,resolve=lambda:actual_path('/shared/group-shared'))
            good='1 0 0:1 / /root rw - gpfs root rw\n2 0 0:2 / /shared/group-shared ro - gpfs shared rw\n'
            def run(text):
                def route(value):
                    if str(value)=='/proc/self/mountinfo':return types.SimpleNamespace(read_text=lambda:text)
                    if str(value)=='/root/group-shared':return alias
                    return actual_path(value)
                with patch.object(entry,'Path',side_effect=route):return entry.ensure_mounts()
            self.assertIn('actual_mounts',run(good))
            for value in (good.replace('gpfs root','overlay root'),good.replace('group-shared ro','group-shared rw'),good+good.splitlines()[0]+'\n'):
                with self.subTest(value=value),self.assertRaises(ValueError):run(value)

    def test_stable_lock_rejects_missing_and_symlink_and_blocks_second_owner(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp).resolve();entry=load('entry',self.strict(root));missing=root/'missing.lock'
            with contextlib.ExitStack()as stack:
                with self.assertRaises(ValueError):entry.acquire(stack,missing,existing=True)
            self.assertFalse(missing.exists());target=root/'own.lock';target.write_text('')
            alias=root/'alias.lock';alias.symlink_to(target)
            with contextlib.ExitStack()as stack:
                with self.assertRaises(ValueError):entry.acquire(stack,alias,existing=True)
            with contextlib.ExitStack()as first,contextlib.ExitStack()as second:
                entry.acquire(first,target,existing=True)
                with self.assertRaises(BlockingIOError):entry.acquire(second,target,existing=True)

    def test_postflight_rejects_fractional_validation_epoch_and_noninteger_checkpoint_epoch(self):
        for mode in ('csv_fraction','checkpoint_float'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as temp:
                root=Path(temp).resolve();strict=self.strict(root);entry=load('entry',strict)
                strict.verify_bundle=lambda:None;strict.snapshot=lambda path:[Path(path).stat().st_size];strict.sha=lambda path:'a'*64
                (strict.RUN/'checks').mkdir(parents=True);(strict.RUN/'ptn/models').mkdir(parents=True)
                for rank in range(8):
                    (strict.RUN/'checks'/('restore_verified_rank%d.json'%rank)).write_text(json.dumps({'schema':'ptn-model47-exact-state-proof/v1','rank':rank,'world_size':8,'model_exact':True,'ordered_optimizer_exact':True,'scaler_exact':True,'contract_sha256':strict.CONTRACT_SHA,'checkpoint_sha256':strict.CK_SHA,'start_epoch':48,'absolute_iteration':705047}))
                    (strict.RUN/'checks'/('first_update_rank%d.json'%rank)).write_text(json.dumps({'rank':rank,'epoch':48,'optimizer_step_completed':True,'absolute_iteration':705047,'contract_sha256':strict.CONTRACT_SHA,'checkpoint_sha256':strict.CK_SHA}))
                    (strict.RUN/'checks'/('first_iteration_rank%d.json'%rank)).write_text(json.dumps({'rank':rank,'absolute_iteration':705047,'lr':.72,'contract_sha256':strict.CONTRACT_SHA,'before_first_forward':True}))
                    for epoch in range(48,121):
                        (strict.RUN/'checks'/('epoch_complete_epoch%d_rank%d.json'%(epoch,rank))).write_text(json.dumps({'rank':rank,'epoch':epoch,'batches_per_rank':15001,'ranks':[{'rank':j,'epoch':epoch,'batches':15001}for j in range(8)],'contract_sha256':strict.CONTRACT_SHA,'before_validation_and_save':True}))
                    folder=strict.RUN/'rank_logs'/str(rank);folder.mkdir(parents=True)
                    for name in ('stdout.log','stderr.log'):(folder/name).write_text('')
                csv='epoch,vox1-eer-8s\n'+''.join(str(epoch+.5 if mode=='csv_fraction' else epoch)+',1.0\n'for epoch in range(48,121))
                (strict.RUN/'ptn/val_log.csv').write_text(csv);(strict.RUN/'ptn/models/model_120.pt').write_text('synthetic')
                fake=types.ModuleType('torch');fake.is_tensor=lambda value:False
                fake.load=lambda *args,**kwargs:{'epoch':120.0 if mode=='checkpoint_float'else 120,'model':{i:0 for i in range(823)},'optimizer':{'state':{i:{}for i in range(584)}},'scaler':{'scale':1.}}
                with patch.dict(sys.modules,{'torch':fake}):
                    with self.assertRaisesRegex(ValueError,'validation epoch'if mode=='csv_fraction'else'final checkpoint'):entry.postflight()

    def test_environment_removes_stage_rank_and_preload_overrides(self):
        with tempfile.TemporaryDirectory()as temp:
            entry=load('entry',self.strict(Path(temp).resolve()))
            with patch.dict(os.environ,{'RANK':'9','WORLD_SIZE':'99','SKIP_AUDIT':'1','LD_PRELOAD':'evil','CHECKPOINT':'bad','TORCHELASTIC_RESTART_COUNT':'5'},clear=False):
                env=entry.environment(['GPU-synthetic'],'fresh-nonce')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-synthetic');self.assertEqual(env['PTN_RESUME_NONCE'],'fresh-nonce')
            for key in ('RANK','WORLD_SIZE','SKIP_AUDIT','LD_PRELOAD','CHECKPOINT','TORCHELASTIC_RESTART_COUNT'):self.assertNotIn(key,env)

    def test_epoch_guard_preserves_actual_signature_and_requires_eight_by_twentyfour(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp).resolve();s=self.strict(root);s.verify_bundle=lambda:None;s.CK=root/'model47';s.sha=lambda path:s.CK_SHA
            guard=load('epoch_guard',s);dist=types.ModuleType('torch.distributed');torch=types.ModuleType('torch');torch.distributed=dist
            audit=types.ModuleType('loader_worker_audit');published=[]
            class Wrapped:
                def __init__(self,*args):self.received=args
                def __iter__(self):return iter(range(15001))
            audit.AuditedLoader=Wrapped;audit.atomic_new=lambda path,value:published.append((path,value))
            def gather(rows,local):
                self.assertEqual(local,{'rank':3,'epoch':48,'batches':15001})
                rows[:]=[{'rank':i,'epoch':48,'batches':15001}for i in range(8)]
            dist.all_gather_object=gather
            for world,n in ((8,24),(8,23),(7,24)):
                with self.subTest(world=world,workers=n):
                    dist.get_world_size=lambda:world;loader=types.SimpleNamespace(num_workers=n)
                    args=[loader,15001,'model','criterion','optimizer','scheduler','margin',48,3,'logger','scaler']
                    def consume(counted,*rest):
                        self.assertEqual(counted.loader.received[:4],(loader,48,3,8));self.assertEqual(rest,tuple(args[1:]))
                        for _ in counted:pass
                        return 'called'
                    original=unittest.mock.Mock(side_effect=consume)
                    with patch.dict(sys.modules,{'torch':torch,'torch.distributed':dist,'loader_worker_audit':audit}):
                        if (world,n)==(8,24):self.assertEqual(guard.guarded_run_epoch(original,*args),'called')
                        else:
                            with self.assertRaises(ValueError):guard.guarded_run_epoch(original,*args)
                            original.assert_not_called()
            self.assertEqual(len(published),1);self.assertTrue(published[0][1]['before_validation_and_save'])

    def test_short_local_or_remote_epoch_is_rejected_before_completion_publication(self):
        for local_count,remote_count in ((15000,15001),(15001,15000)):
            with self.subTest(local=local_count,remote=remote_count),tempfile.TemporaryDirectory()as temp:
                root=Path(temp).resolve();s=self.strict(root);s.verify_bundle=lambda:None;s.CK=root/'ck';s.sha=lambda path:s.CK_SHA
                guard=load('epoch_guard',s);dist=types.ModuleType('torch.distributed');dist.get_world_size=lambda:8
                torch=types.ModuleType('torch');torch.distributed=dist;audit=types.ModuleType('loader_worker_audit')
                audit.AuditedLoader=lambda *args:range(local_count);audit.atomic_new=unittest.mock.Mock()
                def gather(rows,local):
                    self.assertEqual(local['batches'],local_count)
                    rows[:]=[{'rank':i,'epoch':48,'batches':local_count if i==3 else remote_count}for i in range(8)]
                dist.all_gather_object=gather
                def consume(loader,*rest):
                    for _ in loader:pass
                args=[types.SimpleNamespace(num_workers=24),15001,None,None,None,None,None,48,3]
                with patch.dict(sys.modules,{'torch':torch,'torch.distributed':dist,'loader_worker_audit':audit}):
                    with self.assertRaisesRegex(ValueError,'short/inconsistent epoch'):guard.guarded_run_epoch(consume,*args)
                audit.atomic_new.assert_not_called()

    def test_strict_restore_forwards_actual_map_and_concrete_fresh_refs_to_core_gate(self):
        gate=types.ModuleType('state_gate');gate.verify_restored=unittest.mock.Mock(return_value={'synthetic_exact':True})
        spec=importlib.util.spec_from_file_location('runtime_strict_map_probe',RUNTIME/'strict_restore.py');module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'ptn_contracts':C,'state_gate':gate}):spec.loader.exec_module(module)
        module.SPK_SHA='a'*64;module.sha=lambda path:'a'*64;module.CONTRACT_PATH='/new/package/resume-contract.json';module.CONTRACT_SHA='c'*64
        module.CONFIG={'speaker_map':{'path':'/saved/map.json'}}
        refs={'original':{'path':'/control/original.json','sha256':'b'*64},'predecessor':{'path':'/control/predecessor.json','sha256':'d'*64}}
        module.fresh_terminals=lambda:(refs,{})
        result=module.verify_restored(None,None,None,48,15001,8,{},'/new/run/ptn/spk2id.json',scheduler=None,margin=None)
        self.assertEqual(result,{'synthetic_exact':True})
        self.assertEqual(gate.verify_restored.call_args.kwargs['speaker_map_path'],'/new/run/ptn/spk2id.json')
        self.assertEqual(gate.verify_restored.call_args.kwargs['terminal_evidence'],refs)

    def test_cleanup_includes_registered_rank_sessions_after_launcher_exit(self):
        with tempfile.TemporaryDirectory()as temp:
            entry=load('entry',self.strict(Path(temp).resolve()));child=types.SimpleNamespace(pid=100,poll=lambda:0)
            groups={(p,p,10):{'pgid':p,'sid':p,'start_ticks':10}for p in (100,200,300)}
            stop=unittest.mock.Mock();wait=unittest.mock.Mock()
            entry.cleanup_registered(child,groups,'nonce',stop=stop,wait=wait)
            self.assertEqual([c.args[0]['pgid']for c in stop.call_args_list],[200,300,100]);wait.assert_not_called()

    def test_unknown_cleanup_never_repeats_bad_nonce_and_waits_for_two_fresh_absences(self):
        with tempfile.TemporaryDirectory()as temp:
            entry=load('entry',self.strict(Path(temp).resolve()));child=types.SimpleNamespace(pid=100,poll=lambda:None)
            group={'pgid':100,'sid':100,'start_ticks':10};stop=unittest.mock.Mock(side_effect=PermissionError('unknown nonce'))
            observed=iter([None,True,False,False]);waited=[];held=[True]
            def wait(g,c):
                self.assertTrue(held[0]);waited.append(g)
                entry.wait_group_absent(g,c,presence=lambda pg:next(observed),sleep=lambda seconds:self.assertTrue(held[0]))
            entry.cleanup_registered(child,{(100,100,10):group},'nonce',stop=stop,wait=wait)
            stop.assert_called_once();self.assertEqual(waited,[group]);held[0]=False

    def test_ready_is_bound_to_actual_process_environment_and_GPU_FD(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp).resolve();s=self.strict(root);entry=load('entry',s);target=root/'gpu.lock';target.write_text('')
            st=target.stat();nonce='a'*48;uuids=['GPU-'+str(i)for i in range(8)]
            process={'pid':200,'ppid':100,'pgid':200,'sid':200,'start_ticks':30}
            lock={'path':str(target),'uuid':uuids[0],'device':st.st_dev,'inode':st.st_ino,'fd':7}
            value={**process,'group_start_ticks':30,'rank':0,'world_size':8,'nonce':nonce,'contract_sha256':s.CONTRACT_SHA,'package_sha256':s.PACKAGE_SHA,'gpu_lock':lock}
            admission={'GPU':{'UUIDs':uuids,'lock_evidence':[lock]*8}}
            good_env=('PTN_RESUME_NONCE='+nonce+'\0RANK=0\0LOCAL_RANK=0\0WORLD_SIZE=8\0CUDA_VISIBLE_DEVICES='+','.join(uuids)).encode()
            original_path=Path
            for mode in ('good','pid','start','nonce_env','rank_env','inode'):
                actual=json.loads(json.dumps(value));env=good_env
                if mode=='pid':actual['pid']=201
                if mode=='start':actual['start_ticks']=1
                if mode=='nonce_env':env=env.replace(nonce.encode(),b'b'*48)
                if mode=='rank_env':env=env.replace(b'RANK=0',b'RANK=1')
                current=json.loads(json.dumps(admission))
                if mode=='inode':current['GPU']['lock_evidence'][0]['inode']+=1
                def route(path):
                    text=str(path)
                    if text.startswith('/proc/')and text.endswith('/environ'):return types.SimpleNamespace(read_bytes=lambda:env)
                    if text.startswith('/proc/')and '/fd/'in text:return target
                    return original_path(path)
                with self.subTest(mode=mode),patch.object(entry,'Path',side_effect=route),patch.object(entry,'proc_identity',return_value=process),patch.object(entry,'ancestor',return_value=True):
                    if mode=='good':self.assertEqual(entry.validate_ready(actual,0,100,nonce,current),{'pgid':200,'sid':200,'start_ticks':30})
                    else:
                        with self.assertRaises(ValueError):entry.validate_ready(actual,0,100,nonce,current)

    def test_parent_ACK_requires_all_eight_real_registrations_then_fresh_idle(self):
        for missing in (False,True):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory()as temp:
                root=Path(temp).resolve();s=self.strict(root);(s.RUN/'checks').mkdir(parents=True);entry=load('entry',s)
                devices=[{'index':i,'uuid':'GPU-'+str(i),'name':'A800','driver_version':'555.42','total_mib':81920}for i in range(8)]
                admission={'GPU':{'UUIDs':[d['uuid']for d in devices],'snapshots':[{'devices':devices}]}}
                seen=[];poll=iter([None,0]);child=types.SimpleNamespace(pid=100,poll=(lambda:next(poll))if missing else(lambda:None))
                def read(path):
                    rank=int(Path(path).stem.split('rank')[-1])
                    if missing and rank==7:raise entry.PublicationPending('not ready')
                    return {'rank':rank},('%064x'%rank)
                def verified(value,rank,*args):seen.append(rank);return {'pgid':200+rank,'sid':200+rank,'start_ticks':10}
                def fresh():self.assertEqual(len(seen),8);return {'devices':devices}
                with patch.object(entry,'owned_json',side_effect=read),patch.object(entry,'validate_ready',side_effect=verified),patch.object(entry,'gpu_snapshot',side_effect=fresh)as probe,patch.object(entry.time,'sleep'):
                    if missing:
                        with self.assertRaises(ValueError):entry.wait_worker_registration(child,'nonce',admission,{})
                        probe.assert_not_called();self.assertFalse((s.RUN/'checks/cuda_ack.json').exists())
                    else:
                        pins=entry.wait_worker_registration(child,'nonce',admission,{})
                        self.assertEqual(len(pins),8);self.assertEqual(set(json.loads((s.RUN/'checks/cuda_ack.json').read_bytes())['ready_sha256']),{str(i)for i in range(8)})

    def test_worker_cannot_import_torch_or_initialize_CUDA_before_parent_ACK(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp).resolve();s=self.strict(root);s.CONFIG['logical_chain_lock']=str(root/'logical.lock');s.CONFIG['required_existing_locks']=[]
            Path(s.CONFIG['logical_chain_lock']).write_text('');worker=load('worker',s)
            e=types.ModuleType('entry');e.worker_ready_before_cuda=unittest.mock.Mock(side_effect=ValueError('ACK unavailable'))
            import builtins
            original=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name=='torch'or name.startswith('torch.'):raise AssertionError('Torch before ACK')
                return original(name,*args,**kwargs)
            with patch.dict(sys.modules,{'entry':e}),patch.dict(os.environ,{'RANK':'0','LOCAL_RANK':'0','WORLD_SIZE':'8'}),patch.object(builtins,'__import__',side_effect=guarded):
                with self.assertRaisesRegex(ValueError,'ACK unavailable'):worker.run_worker()
            e.worker_ready_before_cuda.assert_called_once_with(0)

    def test_unknown_process_identity_prevents_any_signal(self):
        with tempfile.TemporaryDirectory()as temp:
            entry=load('entry',self.strict(Path(temp).resolve()));child=types.SimpleNamespace(pid=12345,poll=lambda:None)
            with patch.object(entry,'own_members',side_effect=ValueError('unproven nonce')),patch.object(entry.os,'killpg')as kill:
                with self.assertRaises(ValueError):entry.stop_own(child,100,'nonce')
                kill.assert_not_called()

    def test_parent_terminal_change_under_locks_is_rejected(self):
        with tempfile.TemporaryDirectory()as temp:
            s=self.strict(Path(temp).resolve());s.fresh_terminals=lambda:({'original':{'sha256':'b'*64}}, {})
            gate=load('predecessor_gate',s)
            with self.assertRaises(ValueError):gate.verify_unchanged({'refs':{'original':{'sha256':'a'*64}}})

    def test_cleanup_ignores_TERM_and_INT_until_body_finishes_then_restores_both(self):
        with tempfile.TemporaryDirectory()as temp:
            entry=load('entry',self.strict(Path(temp).resolve()));sig=entry.signal
            originals={sig.SIGTERM:object(),sig.SIGINT:object()};current=dict(originals)
            def set_signal(number,handler):
                previous=current[number];current[number]=handler;return previous
            with patch.object(sig,'getsignal',side_effect=lambda number:current[number]),\
                 patch.object(sig,'signal',side_effect=set_signal):
                with self.assertRaisesRegex(ValueError,'synthetic cleanup error'):
                    with entry.cleanup_signal_guard():
                        self.assertIs(current[sig.SIGTERM],sig.SIG_IGN)
                        self.assertIs(current[sig.SIGINT],sig.SIG_IGN)
                        raise ValueError('synthetic cleanup error')
            self.assertEqual(current,originals)
            body=next(n for n in ast.parse((RUNTIME/'entry.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='execute')
            guards=[n for n in ast.walk(body)if isinstance(n,ast.With)and any(
                isinstance(i.context_expr,ast.Call)and isinstance(i.context_expr.func,ast.Name)
                and i.context_expr.func.id=='cleanup_signal_guard'for i in n.items)]
            self.assertEqual(len(guards),1)
            calls={n.func.id for n in ast.walk(guards[0])if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)}
            self.assertTrue({'cleanup_registered','wait_gpu_return'}<=calls)


if __name__=='__main__':unittest.main()
