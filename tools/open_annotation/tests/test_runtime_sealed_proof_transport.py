"""默认关闭的父证明FD传递；合成CPU输入，不加载模型或原音。"""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
import tempfile
import hashlib
import traceback
import fcntl
import subprocess
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'tests')]
import bounded_speed_runtime as M


class DeclarationTests(unittest.TestCase):
    def test_missing_transport_preserves_old_strict_path(self):
        self.assertIsNone(M.proof_transport_profile({'schema':M.NATIVE_SCHEMA}))

    def test_only_explicit_exact_native_transport_is_accepted(self):
        value={'schema':M.NATIVE_SCHEMA,'proof_transport':{'policy':'parent_sealed_memfd/v1'}}
        self.assertEqual(M.proof_transport_profile(value),value['proof_transport'])
        for declaration in (None,True,{'policy':'parent_sealed_memfd/v1','trust_me':True},{'policy':'ordinary_JSON/v1'}):
            with self.subTest(declaration=declaration),self.assertRaises(ValueError):
                M.proof_transport_profile(dict(value,proof_transport=declaration))
        with self.assertRaises(ValueError):M.proof_transport_profile(dict(value,schema=M.SCHEMA))

    def test_plain_dict_or_boolean_never_becomes_selected_capability(self):
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory).resolve()/'runtime.json';p.write_text(json.dumps({'schema':M.NATIVE_SCHEMA,'purpose':'cpu_fixture','proof_transport':M.PROOF_TRANSPORT}))
            pin={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for value in (True,{}, {'selected_verified':True}):
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,'authenticated inherited memfd'):
                    M.compile_runtime(pin,_selected_proof=value)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.file=self.root/'proof.json';self.file.write_bytes(b'original-proof')
        self.pin={'path':str(self.file),'sha256':hashlib.sha256(self.file.read_bytes()).hexdigest()}

    def test_verified_snapshot_deduplicates_paths_and_retains_all_ancestors(self):
        value=M._snapshot_verified_pins([self.pin,self.pin]);self.assertEqual(len(value['files']),1)
        self.assertEqual({d['path']for d in value['directories']},{str(p)for p in self.file.parents})
        M._verify_snapshot(value)

    def test_changed_file_during_hash_cannot_get_a_trusted_after_only_stamp(self):
        original=M.R.pinned_file
        def changed(pin):
            result=original(pin);self.file.write_bytes(b'changed-after-valid-hash');return result
        with patch.object(M.R,'pinned_file',side_effect=changed),self.assertRaisesRegex(ValueError,'around verified SHA'):
            M._snapshot_verified_pins([self.pin])
        self.assertEqual(self.file.read_bytes(),b'changed-after-valid-hash')

    def test_changed_content_even_with_restored_mtime_is_refused_without_self_heal(self):
        value=M._snapshot_verified_pins([self.pin]);before=self.file.stat()
        self.file.write_bytes(b'modified-proof');os.utime(self.file,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'identity/content timestamp|pinned file differs'):M._verify_snapshot(value)
        self.assertEqual(self.file.read_bytes(),b'modified-proof')

    def test_replaced_inode_or_symlink_ancestor_cannot_match_snapshot(self):
        value=M._snapshot_verified_pins([self.pin]);old=self.root/'old.json';self.file.rename(old);self.file.write_bytes(b'original-proof')
        with self.assertRaises(ValueError):M._verify_snapshot(value)
        folder=self.root/'real';folder.mkdir();target=folder/'p.json';target.write_bytes(b'proof')
        link=self.root/'linked';link.symlink_to(folder,target_is_directory=True)
        pin={'path':str(link/'p.json'),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        with self.assertRaisesRegex(ValueError,'nonlinked directory'):M._snapshot_verified_pins([pin])

    def test_snapshot_cannot_drop_or_add_an_ancestor(self):
        value=M._snapshot_verified_pins([self.pin]);value['directories']=value['directories'][1:]
        with self.assertRaisesRegex(ValueError,'ancestor coverage'):M._verify_snapshot(value)

    def test_byte_change_rejected_even_when_all_file_stat_and_fstat_values_are_frozen(self):
        value=M._snapshot_verified_pins([self.pin]);saved_file=os.lstat(self.file)
        saved={str(p):os.lstat(p)for p in [self.file,*self.file.parents]}
        self.file.write_bytes(b'modified-proof')
        lstat0,stat0,fstat0=os.lstat,os.stat,os.fstat
        def known(path):
            if isinstance(path,int):return None
            return saved.get(os.fsdecode(path))
        def frozen_lstat(path,*args,**kwargs):return known(path)or lstat0(path,*args,**kwargs)
        def frozen_stat(path,*args,**kwargs):return known(path)or stat0(path,*args,**kwargs)
        def frozen_fstat(fd):
            actual=fstat0(fd)
            return saved_file if(actual.st_dev,actual.st_ino)==(saved_file.st_dev,saved_file.st_ino)else actual
        with patch.object(os,'stat',side_effect=frozen_stat),patch.object(os,'lstat',side_effect=frozen_lstat),patch.object(os,'fstat',side_effect=frozen_fstat):
            with self.assertRaisesRegex(ValueError,'pinned file differs'):M._verify_snapshot(value)


class SmallControlRoleTests(unittest.TestCase):
    def test_semantic_small_controls_still_hash_but_large_proof_roles_do_not(self):
        from test_native_formal_admission import NativeFixture
        n=NativeFixture(self);r=n.old
        r.runtime.update(schema=M.NATIVE_SCHEMA,new_contract=n.newpin,selected_pilot_admission=n.pin,
                         new_bundle=n.bundle,proof_transport=copy.deepcopy(M.PROOF_TRANSPORT));r.save()
        lease=M.RuntimeLease(r.pin);lease.unique_claim_pin=n.put('small-role-claim.json',{'synthetic_CPU_only':True})
        actual={p['path']for p in lease._small_pins()}
        for role in ('candidate_contract','quality_review','baseline_track','main_track','recovery_track','pilot_result'):
            self.assertIn(n.selected[role]['path'],actual)
        for value in n.selected['resource_evidence'].values():
            for pin in value.values():self.assertIn(pin['path'],actual)
        for role in ('comparison','recovery_comparison'):
            self.assertNotIn(n.selected[role]['path'],actual)
        for pin in n.selected['native_readbacks'].values():self.assertNotIn(pin['path'],actual)

    def test_production_role_enumeration_includes_every_bundle_member_and_seven_SDK_sources_without_weights(self):
        from test_native_formal_admission import NativeFixture
        import native_batch_admission as A
        n=NativeFixture(self,full_code=True)
        selected=A.check_native_selected_pilot(n.pin,n.new,'cpu_fixture')
        contract=copy.deepcopy(n.new);contract['source_files']={};roles={}
        for i in range(14):
            path=n.f.root/('synthetic-SDK-%02d.py'%i);path.write_text('# synthetic CPU source, not model code\n')
            contract['source_files'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for i,role in enumerate(('whisper_feature_extractor','sequence_feature_extractor','batch_feature','processor_mixin','audio_utils','qwen_processor','sdk_normalizer')):
            path=n.f.root/('feature-source-'+role+'.py');path.write_text('# explicit source coverage fixture\n')
            roles[role]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        cfg=n.f.root/'preprocessor_config.json';cfg.write_text('{}\n')
        contract['asr_preprocessing_sources']={'schema':'annotation-official-b1-feature-sources/v1','files':roles,
            'processor_config':{'path':str(cfg),'sha256':hashlib.sha256(cfg.read_bytes()).hexdigest()}}
        compiled={'pins':[],'selected':selected,'new_contract':contract,
            'spec':{'purpose':'production','selected_pilot_admission':n.pin}}
        pins=M._transport_input_pins(compiled);paths={p['path']for p in pins}
        members=json.loads(Path(n.bundle['path']).read_text());self.assertGreater(len(members),1)
        self.assertTrue({str(n.code/name)for name in members}<=paths)
        self.assertTrue(set(contract['source_files'])<=paths)
        self.assertTrue({v['path']for v in roles.values()}<=paths);self.assertIn(str(cfg),paths)
        self.assertFalse(any(p.endswith('.safetensors')for p in paths))


LINUX_MEMFD=sys.platform.startswith('linux')and hasattr(os,'memfd_create')and hasattr(os,'fork')


@unittest.skipUnless(LINUX_MEMFD,'actual Linux memfd/fork/seals tests run separately on CPU-only Linux')
class LinuxTransportTests(unittest.TestCase):
    def setUp(self):
        from test_native_formal_admission import NativeFixture
        self.n=NativeFixture(self);self.r=self.n.old
        self.r.runtime.update(schema=M.NATIVE_SCHEMA,new_contract=self.n.newpin,selected_pilot_admission=self.n.pin,
                              new_bundle=self.n.bundle,proof_transport=copy.deepcopy(M.PROOF_TRANSPORT))
        self.r.save()

    def fork(self,function):
        read,write=os.pipe();pid=os.fork()
        if pid==0:
            os.close(read)
            try:
                value=function();raw=json.dumps({'ok':True,'value':value}).encode()
            except BaseException:raw=json.dumps({'ok':False,'traceback':traceback.format_exc()}).encode()
            os.write(write,raw);os.close(write);os._exit(0)
        os.close(write);raw=b''
        while True:
            block=os.read(read,65536)
            if not block:break
            raw+=block
        os.close(read);_,status=os.waitpid(pid,0);self.assertEqual(os.waitstatus_to_exitcode(status),0)
        value=json.loads(raw);self.assertTrue(value['ok'],value.get('traceback'));return value['value']

    def test_real_fork_inherited_memfd_full_parent_once_and_child_light_lineage(self):
        import native_batch_admission as A
        with patch.object(A,'check_native_selected_pilot',wraps=A.check_native_selected_pilot)as check:
            with M.RuntimeLease(self.r.pin)as parent:
                self.assertEqual(check.call_count,1);evidence=parent.ticket_evidence
                self.assertIn(evidence['selected_proof']['fd'],parent.pass_fds)
                def child():
                    with patch.object(A,'check_native_selected_pilot',side_effect=AssertionError('child may not repeat complete selected readback')):
                        lease=M.RuntimeLease.adopt(self.r.pin,evidence)
                        try:
                            lease.verify();lease.assert_held()
                            return {'purpose':lease.compiled['spec']['purpose'],'sources':lease.compiled['index']['source_count'],
                                    'same_parent':lease.proof.payload['issuer']['pid']==os.getppid(),'no_models':not any(n.split('.')[0]in {'torch','qwen_asr','vllm'}for n in sys.modules)}
                        finally:lease.__exit__(None,None,None)
                result=self.fork(child);self.assertTrue(result['same_parent']);self.assertTrue(result['no_models'])
                self.assertEqual(check.call_count,1)

    def test_real_exec_pass_fds_keeps_sealed_proof_without_inherited_Python_registry(self):
        with M.RuntimeLease(self.r.pin)as parent:
            code="""import sys,json,os
sys.path.insert(0,sys.argv[1])
import bounded_speed_runtime as M
lease=M.RuntimeLease.adopt(json.loads(sys.argv[2]),json.loads(sys.argv[3]))
try:
 lease.verify()
 print(json.dumps({'parent_verified':lease.proof.payload['issuer']['pid']==os.getppid(),'sealed':True,'torch_imported':'torch'in sys.modules}))
finally:lease.__exit__(None,None,None)
"""
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
            result=subprocess.run([sys.executable,'-I','-B','-c',code,str(HERE),json.dumps(self.r.pin),json.dumps(parent.ticket_evidence)],
                pass_fds=parent.pass_fds,env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr);value=json.loads(result.stdout)
            self.assertTrue(value['parent_verified']);self.assertFalse(value['torch_imported'])

    def test_regular_JSON_or_unsealed_memfd_is_never_a_reusable_parent_proof(self):
        with M.RuntimeLease(self.r.pin)as parent:
            good=parent.ticket_evidence;raw=os.pread(parent.proof.fd,M.MAX_PROOF_BYTES,0)
            for kind in ('regular','unsealed'):
                if kind=='regular':
                    path=self.n.f.root/'ordinary-selfsigned-proof.json';path.write_bytes(raw);fd=os.open(path,os.O_RDONLY)
                else:fd=os.memfd_create('unsealed-synthetic',os.MFD_ALLOW_SEALING);os.write(fd,raw)
                try:
                    evidence=copy.deepcopy(good);s=os.fstat(fd)
                    evidence['selected_proof'].update(fd=fd,parent_fd=fd,device=s.st_dev,inode=s.st_ino,bytes=s.st_size)
                    def child():
                        try:M.RuntimeLease.adopt(self.r.pin,evidence)
                        except (ValueError,OSError):return True
                        raise AssertionError('ordinary JSON/unsealed FD accepted')
                    self.assertTrue(self.fork(child))
                finally:os.close(fd)

    def test_wrong_runtime_claim_hash_or_parent_held_fd_is_refused(self):
        with M.RuntimeLease(self.r.pin)as parent:
            good=parent.ticket_evidence
            null=os.open('/dev/null',os.O_RDONLY)
            try:
                for kind in ('hash','claim','parent_fd','runtime'):
                    evidence=copy.deepcopy(good);pin=self.r.pin
                    if kind=='hash':evidence['selected_proof']['sha256']='0'*64
                    elif kind=='claim':evidence['unique_producer_claim']={'path':'/not-this-claim','sha256':'0'*64}
                    elif kind=='parent_fd':evidence['selected_proof']['parent_fd']=null
                    else:pin=dict(self.r.pin,sha256='0'*64);evidence['runtime_spec']=pin
                    def child():
                        try:M.RuntimeLease.adopt(pin,evidence)
                        except (ValueError,OSError):return True
                        raise AssertionError('wrong actual runtime/claim/hash/heldFD accepted')
                    self.assertTrue(self.fork(child))
            finally:os.close(null)

    def test_changed_verified_input_rejects_both_parent_and_child_without_resnapshot(self):
        with M.RuntimeLease(self.r.pin)as parent:
            evidence=parent.ticket_evidence;path=Path(self.n.selected['quality_review']['path']);path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'sealed proof input|pinned file differs'):parent.verify()
            def child():
                try:M.RuntimeLease.adopt(self.r.pin,evidence)
                except ValueError:return True
                raise AssertionError('changed proof was silently resnapshotted')
            self.assertTrue(self.fork(child))

    def test_seals_really_block_write_truncate_and_unsealing(self):
        with M.RuntimeLease(self.r.pin)as parent:
            fd=parent.proof.fd
            with self.assertRaises(OSError):os.pwrite(fd,b'X',0)
            with self.assertRaises(OSError):os.ftruncate(fd,1)
            with self.assertRaises(OSError):fcntl.fcntl(fd,fcntl.F_ADD_SEALS,0)
            parent.verify()

    def test_wrong_issuer_PID_start_or_boot_in_real_sealed_memory_is_refused(self):
        with M.RuntimeLease(self.r.pin)as parent:
            good=parent.ticket_evidence
            for field in ('pid','start_ticks','boot_id'):
                value=copy.deepcopy(parent.proof.payload)
                value['issuer'][field]=('not-this-boot'if field=='boot_id'else value['issuer'][field]+1)
                raw=M.H.P.canonical(value);fd=os.memfd_create(M.MEMFD_NAME,os.MFD_ALLOW_SEALING);os.write(fd,raw)
                fcntl.fcntl(fd,fcntl.F_ADD_SEALS,M._required_seals());s=os.fstat(fd);e=copy.deepcopy(good)
                e['selected_proof'].update(fd=fd,parent_fd=fd,device=s.st_dev,inode=s.st_ino,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                try:
                    def child():
                        try:M.RuntimeLease.adopt(self.r.pin,e)
                        except ValueError:return True
                        raise AssertionError('wrong actual parent identity accepted')
                    self.assertTrue(self.fork(child))
                finally:os.close(fd)

    def test_missing_ancestor_FD_still_rejects_after_correct_selected_proof(self):
        with M.RuntimeLease(self.r.pin)as parent:
            evidence=parent.ticket_evidence;evidence['ancestor_locks']=[]
            def child():
                try:M.RuntimeLease.adopt(self.r.pin,evidence)
                except ValueError:return True
                raise AssertionError('sealed proof bypassed actual ancestor FD checks')
            self.assertTrue(self.fork(child))

    def test_snapshot_cannot_omit_a_real_finite_input_even_with_consistent_ancestor_list(self):
        with M.RuntimeLease(self.r.pin)as parent:
            value=copy.deepcopy(parent.proof.payload);snapshot=value['snapshot']
            removed=next(i for i,v in enumerate(snapshot['files'])if 'batches'in Path(v['pin']['path']).parts)
            snapshot['files'].pop(removed)
            needed=sorted({str(p)for item in snapshot['files']for p in Path(item['pin']['path']).parents})
            indexed={d['path']:d for d in snapshot['directories']};snapshot['directories']=[indexed[p]for p in needed]
            raw=M.H.P.canonical(value);fd=os.memfd_create(M.MEMFD_NAME,os.MFD_ALLOW_SEALING);os.write(fd,raw)
            fcntl.fcntl(fd,fcntl.F_ADD_SEALS,M._required_seals());s=os.fstat(fd);e=copy.deepcopy(parent.ticket_evidence)
            e['selected_proof'].update(fd=fd,parent_fd=fd,device=s.st_dev,inode=s.st_ino,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            try:
                def child():
                    try:M.RuntimeLease.adopt(self.r.pin,e)
                    except ValueError:return True
                    raise AssertionError('incomplete but self-consistent finite snapshot accepted')
                self.assertTrue(self.fork(child))
            finally:os.close(fd)

    def test_repeated_parent_verification_hashes_every_finite_input_and_small_critical_role(self):
        with M.RuntimeLease(self.r.pin)as parent:
            small={p['path']for p in parent._small_pins()};large={v['pin']['path']for v in parent.proof.payload['snapshot']['files']}-small
            called=[];original=M.R.file_sha;open0=os.open;read0=os.read;fds={};read_bytes={};read_EOFs={}
            def observed(path):
                called.append(str(path))
                return original(path)
            def opening(path,*args,**kwargs):
                fd=open0(path,*args,**kwargs);fds[fd]=str(path);return fd
            def reading(fd,size):
                block=read0(fd,size);path=fds[fd]
                read_bytes[path]=read_bytes.get(path,0)+len(block)
                if not block:read_EOFs[path]=read_EOFs.get(path,0)+1
                return block
            with patch.object(M.R,'file_sha',side_effect=observed),patch.object(os,'open',side_effect=opening),patch.object(os,'read',side_effect=reading):
                for _ in range(3):parent.verify()
            self.assertIn(self.n.selected['quality_review']['path'],called)
            self.assertIn(self.n.selected['candidate_contract']['path'],called)
            self.assertTrue(large);self.assertTrue(large<=set(read_EOFs))
            for path in large:
                self.assertGreaterEqual(read_EOFs[path],3)
                self.assertGreaterEqual(read_bytes[path],3*Path(path).stat().st_size)

    def test_current_platform_TTL_and_old_lineage_compile_are_not_skipped(self):
        with M.RuntimeLease(self.r.pin)as parent:
            evidence=parent.ticket_evidence
            def child():
                with patch.object(M,'check_platform',side_effect=ValueError('real light platform/TTL path called')):
                    try:M.RuntimeLease.adopt(self.r.pin,evidence)
                    except ValueError as error:
                        if 'light platform/TTL'in str(error):return True
                        raise
                    raise AssertionError('platform/old-lineage compilation skipped')
            self.assertTrue(self.fork(child))

    def test_default_native_without_declaration_keeps_complete_checks(self):
        self.r.runtime.pop('proof_transport');self.r.save()
        import native_batch_admission as A
        with patch.object(A,'check_native_selected_pilot',wraps=A.check_native_selected_pilot)as check:
            with M.RuntimeLease(self.r.pin)as parent:
                self.assertIsNone(parent.proof);self.assertNotIn('selected_proof',parent.ticket_evidence)
                self.assertEqual(check.call_count,1)
                child=M.RuntimeLease.adopt(self.r.pin,parent.ticket_evidence)
                try:self.assertEqual(check.call_count,2)
                finally:child.__exit__(None,None,None)


if __name__=='__main__':unittest.main()
