"""Known synthetic /proc tree; no Torch, real signals, models or GPU queries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from tools.ptn_resume import live_process_probe as L


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);raw=(json.dumps(value,sort_keys=True)+'\n').encode();path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def proc_stat(root,pid,ppid,pgid,sid,start,state='S'):
    path=root/str(pid)/'stat';path.parent.mkdir(parents=True,exist_ok=True)
    values=[state,str(ppid),str(pgid),str(sid)]+['0']*15+[str(start)]
    path.write_text('%d (synthetic ) name) '%pid+' '.join(values)+'\n')


class LiveProbeTests(unittest.TestCase):
    def fixture(self,root):
        run=root/'run';checks=run/'checks';proc=root/'proc';package=root/'package';(package/'source').mkdir(parents=True)
        gate=package/'runtime/state_gate.py';gate.parent.mkdir();gate.write_text('# synthetic gate, never imported\n');gate_sha=hashlib.sha256(gate.read_bytes()).hexdigest()
        c=write(package/'resume-contract.json',{'schema':'ptn-resume-model47/v1','new_run_root':str(run),'source_root':str(package/'source'),'runtime_files':{'state_gate.py':gate_sha}})
        p=write(package/'bundle_files.json',{'resume-contract.json':c,'runtime/state_gate.py':gate_sha})
        nonce='c'*48;uuids=['GPU-synthetic-%d'%i for i in range(8)]
        proc_stat(proc,50,1,50,50,10);locks=[];gathered=[]
        for rank in range(8):
            pid=100+rank;start=1000+rank;proc_stat(proc,pid,50,pid,pid,start)
            target=root/'locks'/('gpu%d.lock'%rank);target.parent.mkdir(exist_ok=True);target.write_text('');st=target.stat()
            lock={'path':str(target),'uuid':uuids[rank],'fd':7,'device':st.st_dev,'inode':st.st_ino};locks.append(lock)
            fd=proc/str(pid)/'fd/7';fd.parent.mkdir();fd.symlink_to(target)
            env={'PTN_RESUME_NONCE':nonce,'PTN_BUNDLE_MANIFEST_SHA':p,'RANK':str(rank),'LOCAL_RANK':str(rank),'WORLD_SIZE':'8','CUDA_VISIBLE_DEVICES':','.join(uuids),'PRIVATE_NOT_EXPORTED':'not-a-credential'}
            (proc/str(pid)/'environ').write_bytes(b'\0'.join((k+'='+v).encode()for k,v in env.items()))
            ready={'rank':rank,'world_size':8,'nonce':nonce,'contract_sha256':c,'package_sha256':p,'pid':pid,'ppid':50,'pgid':pid,'sid':pid,'start_ticks':start,'group_start_ticks':start,'gpu_lock':lock}
            write(checks/('worker_ready_rank%d.json'%rank),ready)
            write(checks/('restore_verified_rank%d.json'%rank),{'schema':'ptn-model47-exact-state-proof/v1','rank':rank,'world_size':8,'contract_sha256':c,'model_exact':True,'ordered_optimizer_exact':True,'scaler_exact':True,'state_gate_sha256':gate_sha})
            proofs=[];localnonce=('%048x'%(rank+1))
            for worker in range(24):
                child=1000+rank*24+worker;childstart=2000+rank*24+worker;proc_stat(proc,child,pid,pid,pid,childstart)
                value={'schema':'ptn-loader-worker-audit/v1','rank':rank,'worker_id':worker,'epoch':48,'world_size':8,'actual_dist_world_size':8,'sampler_inherited_world_size':8,'num_workers':24,'actual_dist_rank':rank,'sampler_inherited_rank':rank,'sampler_epoch':48,'nonce':localnonce,'bundle_sha256':p,'ppid':pid,'pid':child,'start_ticks':childstart,'actual_context':'fork','actual_dist_initialized':True,'sampler_partition':True,'legacy_aug_behavior':True,'original_seed_worker_called':True,'worker_collective_called':False,'worker_GPU_operation_called':False}
                path=checks/'loader_workers/epoch48'/('rank%d'%rank)/('worker%d.json'%worker);pin=write(path,value)
                proofs.append({'worker_id':worker,'path':str(path),'sha256':pin,'pid':child})
            gathered.append({'nonce':localnonce,'parent_rank_pid':pid,'rank':rank,'world_size':8,'epoch':48,'worker_count':24,'expected_workers':24,'error':None,'proofs':proofs})
        for rank in range(8):write(checks/'loader_workers'/('gate_epoch48_rank%d.json'%rank),{'schema':'ptn-loader-gate/v1','rank':rank,'epoch':48,'world_size':8,'workers_per_rank':24,'global_verified_workers':192,'before_first_batch_yield':True,'ranks':gathered})
        write(checks/'parent_admission.json',{'contract_sha256':c,'package_manifest_sha256':p,'nonce':nonce,'GPU':{'UUIDs':uuids,'lock_evidence':locks},'parent':{'pid':50,'start_ticks':10},'source':{'source':str(package/'source')}})
        return {'run_root':run,'contract_sha':c,'package_sha':p,'expected_hostname':socket.gethostname(),'proc':proc,
                'gpu_reader':lambda:{'devices':[{'index':i,'uuid':u}for i,u in enumerate(uuids)],'raw_queries':[],'compute_PID_is_not_container_PID_translation':True}}

    def test_complete_200_identities_and_only_eight_known_rank_environments(self):
        with tempfile.TemporaryDirectory()as d:
            a=self.fixture(Path(d).resolve());r=L.probe(**a)
            self.assertTrue(r['passed']);self.assertEqual(r['unique_rank_and_loader_PID_count'],200)
            self.assertEqual(r['rank_environments_read'],8);self.assertEqual(r['loader_environments_read'],0)
            self.assertFalse(r['proc_directory_enumerated']);self.assertFalse(r['CUDA_initialized'])
            self.assertNotIn('PRIVATE_NOT_EXPORTED',json.dumps(r))

    def test_wrong_host_or_stale_rank_start_or_nonce_refused(self):
        for mode in ('host','start','nonce'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as d:
                a=self.fixture(Path(d).resolve())
                if mode=='host':a['expected_hostname']='different-cloud'
                if mode=='start':proc_stat(a['proc'],100,50,100,100,9000)
                if mode=='nonce':(a['proc']/'100/environ').write_bytes(b'PTN_RESUME_NONCE=wrong')
                with self.assertRaises(ValueError):L.probe(**a)

    def test_zombie_reparented_or_missing_loader_refused(self):
        for mode in ('zombie','reparent','missing'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as d:
                a=self.fixture(Path(d).resolve())
                if mode=='zombie':proc_stat(a['proc'],1000,100,100,100,2000,'Z')
                if mode=='reparent':proc_stat(a['proc'],1000,999,100,100,2000)
                if mode=='missing':(a['proc']/'1000/stat').unlink()
                with self.assertRaises((ValueError,FileNotFoundError)):L.probe(**a)

    def test_changed_worker_SHA_or_GPU_fd_refused(self):
        for mode in ('sha','fd'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as d:
                a=self.fixture(Path(d).resolve())
                if mode=='sha':(a['run_root']/'checks/loader_workers/epoch48/rank0/worker0.json').write_text('{}')
                else:(a['proc']/'100/fd/7').unlink()
                with self.assertRaises((ValueError,FileNotFoundError)):L.probe(**a)

    def test_duplicate_consistently_resealed_loader_PID_refused(self):
        with tempfile.TemporaryDirectory()as d:
            a=self.fixture(Path(d).resolve());checks=a['run_root']/'checks';path=checks/'loader_workers/epoch48/rank0/worker1.json'
            v=json.loads(path.read_bytes());v.update(pid=1000,start_ticks=2000);pin=write(path,v)
            for rank in range(8):
                gate=checks/'loader_workers'/('gate_epoch48_rank%d.json'%rank);g=json.loads(gate.read_bytes());g['ranks'][0]['proofs'][1].update(pid=1000,sha256=pin);write(gate,g)
            with self.assertRaisesRegex(ValueError,'duplicate'):L.probe(**a)

    def test_CLI_refuses_RUN_package_existing_and_dangling_output_before_probe(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d).resolve();a=self.fixture(root);existing=root/'existing.json';existing.write_text('preserve');link=root/'link';link.symlink_to(root/'absent')
            for output in (a['run_root']/'new.json',root/'package/new.json',existing,link):
                args=['--run-root',str(a['run_root']),'--contract-sha',a['contract_sha'],'--package-sha',a['package_sha'],'--expected-hostname',socket.gethostname(),'--output',str(output)]
                with self.subTest(output=output),patch.object(L,'probe',side_effect=AssertionError('not admitted'))as probe:
                    with self.assertRaises(ValueError):L.main(args)
                    probe.assert_not_called()
            self.assertEqual(existing.read_text(),'preserve');self.assertFalse((root/'absent').exists())

    def test_normal_R_to_S_scheduling_change_is_not_identity_change(self):
        with tempfile.TemporaryDirectory()as d:
            a=self.fixture(Path(d).resolve());original=L.proc_identity;calls={}
            def observed(pid,proc):
                value=original(pid,proc);calls[pid]=calls.get(pid,0)+1
                value['state']='R'if calls[pid]==1 else'S'
                return value
            with patch.object(L,'proc_identity',side_effect=observed):r=L.probe(**a)
            self.assertTrue(r['passed']);self.assertEqual(r['scheduling_states_at_two_reads']['1000'],{'before':'R','after':'S'})

    def test_epoch_transition_failure_is_structured_indeterminate_not_passed(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d).resolve();a=self.fixture(root);output=root/'observation.json'
            args=['--run-root',str(a['run_root']),'--contract-sha',a['contract_sha'],'--package-sha',a['package_sha'],'--expected-hostname',socket.gethostname(),'--output',str(output)]
            with patch.object(L,'probe',side_effect=FileNotFoundError('worker exited at epoch transition')):code=L.main(args)
            self.assertEqual(code,2);r=json.loads(output.read_bytes());self.assertFalse(r['passed']);self.assertEqual(r['status'],'indeterminate')


if __name__=='__main__':unittest.main()
