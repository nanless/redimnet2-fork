"""提速新生产者的有界队列：原片懒读、旧结果原producer保留、无数据库。

由新冻结worker显式选择；不会修改旧queue或旧labels。资源控制器要
把ExistingProducerLease.pass_fds传入每个GPU子进程，并在CUDA排空后关闭。
"""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import datetime
import json
import os
from pathlib import Path
import secrets
import stat
import struct

if __package__:
    from . import bounded_speed_handoff as H
    from .contracts import Refused
else:
    import bounded_speed_handoff as H
    from contracts import Refused
P = H.P
need = H.need


class AllShardsQuarantined(Refused):
    """所有未完成片都已隔离：显式失败并留证据，不再出现空候选的尾部无限循环。"""


def write_new(path, value):
    path = P.physical(path);path.parent.mkdir(parents=True,exist_ok=True)
    return P.write_new(path, value)


def replace_json(path, value):
    path = P.physical(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name('.'+path.name+'.'+secrets.token_hex(16)+'.tmp')
    P.write_new(tmp, value)
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


@contextlib.contextmanager
def new_lock(path, nonblocking=False):
    path=P.physical(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
    stream=os.fdopen(fd,'r+b')
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|(fcntl.LOCK_NB if nonblocking else 0))
        before=os.fstat(fd);after=path.stat()
        need(stat.S_ISREG(before.st_mode)and(before.st_dev,before.st_ino)==(after.st_dev,after.st_ino),'new lock inode differs')
        yield stream
    finally:stream.close()


class ExistingProducerLease:
    """旧writer终态＋已有锁在新GPU整个生命周期内持有；不创建替代旧锁。"""
    def __init__(self,index_path,index_sha,terminal_pin):
        self.reader=H.BoundedShardReader(index_path,index_sha)
        self.index_path=Path(index_path);self.index_sha=index_sha
        self.index=self.reader.index;self.terminal_pin=terminal_pin
        self._handles=[];self.held=False

    @property
    def pass_fds(self):
        self.assert_held();return tuple(h['fd']for h in self._handles)

    @property
    def fd_evidence(self):
        self.assert_held()
        return [dict(path=str(h['path']),fd=h['fd'],device=h['device'],inode=h['inode'])for h in self._handles]

    @classmethod
    def adopt_inherited(cls,index_path,index_sha,terminal_pin,fd_evidence):
        """仅供已由父ticket绑定的GPU子进程复用同一打开描述，不重开旧锁。"""
        obj=cls(index_path,index_sha,terminal_pin)
        q=P.physical(Path(obj.index['old_queue_root']))
        obj.terminal=P.terminal_observation(Path(terminal_pin['path']),terminal_pin['sha256'],obj.index['old_task_id'],q)
        expected={str(q/'.parent.lock')}|{str(P.child(q,'claims/'+sid+'.lock'))for sid in obj.index['parent_lease_shard_ids']}
        need(isinstance(fd_evidence,list)and len(fd_evidence)==len(expected)and {v['path']for v in fd_evidence}==expected and len({v['fd']for v in fd_evidence})==len(expected),'inherited old lock set differs')
        try:
            for v in fd_evidence:
                fd=v['fd'];need(type(fd)is int and fd>2,'explicit inherited lock FD required')
                current=os.fstat(fd);path=P.physical(Path(v['path']));entry=path.stat()
                need(stat.S_ISREG(current.st_mode)and(current.st_dev,current.st_ino)==(entry.st_dev,entry.st_ino)==(v['device'],v['inode']),'inherited old lock FD identity differs')
                if obj.index['purpose']=='production':
                    info=Path('/proc/self/fdinfo')/str(fd)
                    need(info.is_file() and any('FLOCK' in line and 'WRITE' in line for line in info.read_text().splitlines() if line.startswith('lock:')),'inherited old lock ownership not visible')
                dup=os.dup(fd)
                obj._handles.append(dict(path=path,fd=dup,device=current.st_dev,inode=current.st_ino))
            obj.held=True
            obj.verify_frozen_state()
            return obj
        except BaseException:
            obj.__exit__(None,None,None)
            raise

    def __enter__(self):
        queue=P.physical(Path(self.index['old_queue_root']))
        self.terminal=P.terminal_observation(Path(self.terminal_pin['path']),self.terminal_pin['sha256'],self.index['old_task_id'],queue)
        try:
            paths=[queue/'.parent.lock']+[P.child(queue,'claims/'+sid+'.lock')for sid in self.index['parent_lease_shard_ids']]
            for path in paths:
                path=P.physical(path);fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
                before=os.fstat(fd)
                self._handles.append(dict(path=path,fd=fd,device=before.st_dev,inode=before.st_ino))
                need(stat.S_ISREG(before.st_mode),'old lock is not regular')
                try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError as exc:raise ValueError('old writer/claim lock still held')from exc
            self.held=True
            self.verify_frozen_state()
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def assert_held(self):
        need(self.held and self._handles,'old writer lease is not held')
        for h in self._handles:
            before=os.fstat(h['fd']);after=P.physical(h['path']).stat()
            need((before.st_dev,before.st_ino)==(after.st_dev,after.st_ino)==(h['device'],h['inode']),'old lock inode replaced')

    def verify_frozen_state(self):
        """仅在GPU前及交接边界调用；不为每条来源重读完整索引。"""
        self.assert_held()
        H.read_pin({'path':str(self.index_path),'sha256':self.index_sha})
        parent,_=H.read_pin(self.index['parent_index'])
        contract,_=H.read_pin(self.index['old_contract'])
        need(contract.get('version')==self.index['old_contract_version']==Path(self.index['old_sidecar_tree']).name,'old contract changed/version differs')
        q=Path(self.index['old_queue_root']);broker,raw=P.read_json(q/'broker.json',1024**2)
        need(P.byte_sha(raw)==self.index['broker_snapshot_sha256'],'old broker changed after handoff plan')
        need(broker['next']==self.index['parent_cursor']and sorted(broker['leases'])==self.index['parent_lease_shard_ids'],'old broker cursor/leases differ from thin index')
        _,identity_raw=P.read_json(q/'identity.json',65536)
        need(P.byte_sha(identity_raw)==self.index['old_queue_identity_sha256'],'old queue identity changed')
        slots=H.capture_slots(q,parent['shards'],broker)
        need(P.byte_sha(slots)==self.index['retained_completed_slots_sha256'],'old completed slot bytes changed')
        return True

    def __exit__(self,*args):
        self.held=False
        for h in reversed(self._handles):os.close(h['fd'])
        self._handles=[]


class BoundedWorkQueue:
    """新queue只消耗薄索引；混合结果显式逐源保留原生产者身份。"""
    def __init__(self,index_path,index_sha,root,contract_sha,new_sidecar_tree,old_lease):
        need(H.is_sha(contract_sha),'new contract SHA required')
        old_lease.assert_held()
        need(old_lease.index_sha==index_sha and old_lease.index_path==Path(index_path),'wrong old writer lease')
        self.old_lease=old_lease;self.reader=old_lease.reader
        self.index_path=Path(index_path);self.index_sha=index_sha;self.index=self.reader.index
        self.shards=self.index['shards'];self.by_id={s['id']:s for s in self.shards};self.ordinal={s['id']:i for i,s in enumerate(self.shards)}
        need(bool(self.shards),'no missing sources: do not create successor GPU task')
        self.root=P.physical(Path(root));self.contract_sha=contract_sha
        self.old_contract_sha=self.index['old_contract']['sha256'];self.old_tree=P.physical(Path(self.index['old_sidecar_tree']))
        self.new_tree=P.physical(Path(new_sidecar_tree))
        self.portable=H.load_portable_context(self.index,self.index['old_contract_version'])
        need(contract_sha!=self.old_contract_sha,'speed successor needs explicit new contract identity')
        need(str(self.root)==self.index['new_queue_root'],'new queue root differs from handoff plan')
        base=P.physical(Path(self.index['new_sidecar_root']))
        need(base in self.new_tree.parents and self.new_tree!=self.old_tree,'new sidecar tree outside explicit new root')
        self.root.mkdir(parents=True,exist_ok=True)
        self.identity=dict(schema='redimnet2-bounded-work-queue/v1',index_sha256=index_sha,contract_sha256=contract_sha,
                           old_contract_sha256=self.old_contract_sha,source_count=self.index['source_count'],shard_count=len(self.shards))
        with new_lock(self.root/'broker.lock'):
            p=self.root/'identity.json'
            if p.exists():need(P.read_json(p,65536)[0]==self.identity,'new queue belongs to another handoff/contract')
            else:write_new(p,self.identity)
            # broker.json 是游标与完成计数的唯一权威；缺失时不静默重建为 next=0。
            # 只有“完全没有任何进度证据”的全新队列才允许初始化一次。
            broker=self.root/'broker.json'
            if not broker.exists():
                evidence=[path for path in (self.root/'completion_map.bin',self.root/'completed',
                                            self.root/'claims',self.root/'attempt_state')
                          if path.exists() and (not path.is_dir() or any(path.iterdir()))]
                if evidence:
                    raise Refused('broker.json missing under %s; refusing to rebuild a zero cursor. '
                        '恢复指引: 从 queue/ 的同级备份或 index+已完成 receipt 重建 broker，'
                        '并用 tools/open_annotation/bounded_metadata_progress.py 交叉核对 completed_chunks 后再启动'
                        % self.root)
                replace_json(broker,dict(next=0,leases={},completed_chunks=0))
            p=self.root/'completion_map.bin'
            if not p.exists():
                with p.open('xb')as f:f.truncate(len(self.shards)*1024);f.flush();os.fsync(f.fileno())
            facts=p.lstat()
            need(stat.S_ISREG(facts.st_mode)and facts.st_nlink==1 and facts.st_size==len(self.shards)*1024,'new completion map shape/type/link differs')

    def chunk_receipt_path(self,sid):return P.child(self.root,'completed/'+sid+'.json')

    def _normalize_receipt(self,source,value):
        value=dict(value)
        producer=value.setdefault('producer_contract_sha256',self.contract_sha)
        tree=value.setdefault('producer_sidecar_tree',str(self.new_tree))
        retained=value.setdefault('retained_existing',False)
        need(producer in (self.contract_sha,self.old_contract_sha),'receipt has undeclared producer')
        need((producer==self.old_contract_sha and tree==str(self.old_tree) and retained is True)or
             (producer==self.contract_sha and tree==str(self.new_tree) and retained is False),'receipt producer tree/retention differs')
        need(value.get('source_id')==source['source_id'] and value.get('job_key')==P.old_job_key(source,producer),'receipt source/job key differs')
        need(value.get('sidecar_relpath')==source['sidecar_relpath'],'receipt sidecar path differs')
        need(value.get('decision')in ('candidate','unknown','failed')and H.is_sha(value.get('sidecar_sha256'))and
             H.is_sha(value.get('commit_sha256'))and type(value.get('sidecar_bytes'))is int and value['sidecar_bytes']>0,'receipt missing actual output identity')
        if producer==self.old_contract_sha and self.portable:
            record,raw=P.read_json(P.child(self.old_tree,source['sidecar_relpath']),32*1024**2)
            _,marker_raw=P.read_json(P.child(self.old_tree,source['sidecar_relpath']+'.commit.json'),65536)
            need(P.byte_sha(raw)==value['sidecar_sha256']and P.byte_sha(marker_raw)==value['commit_sha256'],'retained old receipt bytes differ')
            proof=value.get('portable_source_evidence')
            need(isinstance(proof,dict)and proof.get('sidecar_sha256')==value['sidecar_sha256']and proof.get('commit_sha256')==value['commit_sha256'],'retained receipt requires bound actual portable proof')
            self.portable['module'].module.validate_ledger_entry(proof,source,record,self.old_contract_sha,self.portable['spec'],spec_sha=self.portable['spec_sha256'])
        return value

    def previous_receipt(self,claim,source):
        claim.assert_current();claim.assert_member(source)
        ref=self.reader.lookup_previous_for_claim(claim.shard['id'],source)
        if ref is None:return None
        value=dict(source_id=source['source_id'],job_key=P.old_job_key(source,self.old_contract_sha),sidecar_relpath=source['sidecar_relpath'],
                   sidecar_sha256=ref['sidecar_sha256'],sidecar_bytes=Path(ref['sidecar_path']).stat().st_size,
                   commit_sha256=ref['commit_sha256'],decision=ref['decision'],producer_contract_sha256=self.old_contract_sha,
                   producer_sidecar_tree=str(self.old_tree),retained_existing=True)
        if 'portable_source_evidence'in ref:value['portable_source_evidence']=ref['portable_source_evidence']
        return self._normalize_receipt(source,value)

    def receipt(self,sid):
        path=self.chunk_receipt_path(sid)
        if not path.exists():return None
        value,_=P.read_json(path,32*1024**2);shard=self.by_id[sid]
        need(value.get('schema')=='redimnet2-bounded-chunk-commit/v1'and value.get('shard_id')==sid and
             value.get('index_sha256')==self.index_sha and value.get('contract_sha256')==self.contract_sha and
             value.get('shard_sha256')==shard['sha256']and value.get('source_count')==shard['count'],'new chunk receipt identity differs')
        sources={s['source_id']:s for s in self.reader.read(sid)};receipts=value['sources']
        need(len(receipts)==len(sources)and{r['source_id']for r in receipts}==set(sources),'new chunk receipt source denominator differs')
        for r in receipts:need(self._normalize_receipt(sources[r['source_id']],r)==r,'receipt omitted explicit producer fields')
        return value

    def write_completion(self,sid,receipt):
        value=dict(shard_id=sid,shard_sha256=self.by_id[sid]['sha256'],source_count=receipt['source_count'],
                   source_ids_sha256=P.digest(sorted(r['source_id']for r in receipt['sources'])),
                   commit_sha256=P.file_sha(self.chunk_receipt_path(sid)),index_sha256=self.index_sha,contract_sha256=self.contract_sha,
                   decision_counts={k:sum(r['decision']==k for r in receipt['sources'])for k in ('candidate','unknown','failed')},
                   retained_source_count=sum(r['retained_existing']for r in receipt['sources']),
                   current_producer_source_count=sum(not r['retained_existing']for r in receipt['sources']),
                   retained_producer_contract_sha256=self.old_contract_sha,
                   current_producer_decision_counts={k:sum(r['decision']==k and not r['retained_existing']for r in receipt['sources'])for k in ('candidate','unknown','failed')},
                   retained_decision_counts={k:sum(r['decision']==k and r['retained_existing']for r in receipt['sources'])for k in ('candidate','unknown','failed')})
        raw=P.canonical(value);need(len(raw)<=980,'completion slot exceeds fixed bound')
        data=b'RDMAP001'+struct.pack('>I',len(raw))+hashlib.sha256(raw).digest()+raw+bytes(980-len(raw))
        fd=os.open(self.root/'completion_map.bin',os.O_RDWR)
        try:
            start=self.ordinal[sid]*1024;written=0
            while written<len(data):
                n=os.pwrite(fd,data[written:],start+written);need(n>0,'short slot write');written+=n
            os.fsync(fd);need(os.pread(fd,1024,start)==data,'new completion readback differs')
        finally:os.close(fd)
        return value

    QUARANTINE_FAILURES=3

    def quarantine_path(self):return self.root/'quarantine.json'

    def _quarantine(self):
        path=self.quarantine_path()
        if not path.exists():return {}
        value=P.read_json(path,1024**2)[0]
        need(isinstance(value,dict)and all(isinstance(entry,dict)for entry in value.values()),'quarantine file is malformed')
        return value

    def quarantine(self):
        """只读隔离快照；不构造 claim、不改 broker。"""
        return self._quarantine()

    def _quarantine_shard(self,quarantine,sid,failures,reason):
        entry=dict(quarantine.get(sid)or{})
        entry.update(quarantined=True,failures=max(int(entry.get('failures')or 0),failures),
                     last_error=str(reason)[:400],quarantined_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        value=dict(quarantine);value[sid]=entry
        replace_json(self.quarantine_path(),value)
        quarantine.clear();quarantine.update(value)
        return entry

    def claim_next(self,worker_id):
        self.old_lease.assert_held()
        with new_lock(self.root/'broker.lock'):
            state=P.read_json(self.root/'broker.json',1024**2)[0]
            need(type(state.get('next'))is int and 0<=state['next']<=len(self.shards) and isinstance(state.get('leases'),dict),'bad new broker cursor')
            need(all(sid in self.by_id and self.ordinal[sid]<state['next']for sid in state['leases']),'new lease outside cursor')
            need(type(state.get('completed_chunks'))is int and state['completed_chunks']==state['next']-len(state['leases']),'new broker completed count differs')
            quarantine=self._quarantine()
            candidates=sorted(state['leases'],key=self.ordinal.get)
            if state['next']<len(self.shards):candidates.append(self.shards[state['next']]['id'])
            offered=0
            for sid in candidates:
                if (quarantine.get(sid)or{}).get('quarantined'):continue
                offered+=1
                lock=P.child(self.root,'claims/'+sid+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
                need(not lock.is_symlink(),'new claim symlink refused')
                fd=os.open(lock,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0),0o600);handle=os.fdopen(fd,'r+b')
                facts=os.fstat(fd);entry=lock.stat()
                if not(stat.S_ISREG(facts.st_mode)and facts.st_nlink==1 and(facts.st_dev,facts.st_ino)==(entry.st_dev,entry.st_ino)):
                    handle.close();raise ValueError('new claim lock type/link/inode differs')
                try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:handle.close();continue
                try:
                    p=P.child(self.root,'attempt_state/'+sid+'.json')
                    generation=P.read_json(p,65536)[0]['generation']+1 if p.exists()else 1
                    token=dict(generation=generation,nonce=secrets.token_hex(24),worker_id=str(worker_id),pid=os.getpid(),claimed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    replace_json(p,token)
                    if state['next']<len(self.shards)and self.shards[state['next']]['id']==sid:state['next']+=1
                    state['leases'][sid]=token;replace_json(self.root/'broker.json',state)
                    # 先落租约与代数：毒片的 receipt 校验失败也要留下可累计的失败次数。
                    try:previous=self.receipt(sid)
                    except (ValueError,Refused) as exc:
                        if generation>self.QUARANTINE_FAILURES:
                            self._quarantine_shard(quarantine,sid,generation-1,'chunk receipt refused %d times: %s'%(generation-1,str(exc)[:300]))
                            handle.close();continue
                        raise
                    if previous is not None:
                        self.write_completion(sid,previous)
                        if sid in state['leases']:state['leases'].pop(sid);state['completed_chunks']+=1
                        replace_json(self.root/'broker.json',state);handle.close();continue
                    if generation>self.QUARANTINE_FAILURES:
                        self._quarantine_shard(quarantine,sid,generation-1,'shard reclaimed %d times without a committed chunk receipt'%(generation-1))
                        handle.close();continue
                    return BoundedClaim(self,self.by_id[sid],handle,token)
                except BaseException:handle.close();raise
            if not offered:
                if state['next']>=len(self.shards)and not state['leases']:return None
                blocked=sorted(sid for sid,entry in quarantine.items()if (entry or {}).get('quarantined'))
                raise AllShardsQuarantined('every unfinished shard is quarantined: '+','.join(blocked[:20])+
                    '; inspect attempts/<shard> and completed/<shard>.json before clearing quarantine.json')
        return None

    def is_complete(self):
        self.old_lease.assert_held()
        with new_lock(self.root/'broker.lock'):
            s=P.read_json(self.root/'broker.json',1024**2)[0]
            return s['completed_chunks']==len(self.shards)and not s['leases']


class BoundedClaim:
    def __init__(self,queue,shard,handle,token):
        self.queue=queue;self.shard=shard;self.handle=handle;self.token=token;self.closed=False
        self.attempt_id='%08d-%s'%(token['generation'],token['nonce'])
        self.attempt_root=P.child(queue.root,'attempts/'+shard['id']+'/'+self.attempt_id);self.attempt_root.mkdir(parents=True,exist_ok=False)
        write_new(self.attempt_root/'claim.json',token)

    def assert_current(self):
        self.queue.old_lease.assert_held();need(not self.closed and not self.handle.closed,'new claim already released')
        before=os.fstat(self.handle.fileno());after=P.child(self.queue.root,'claims/'+self.shard['id']+'.lock').stat()
        need((before.st_dev,before.st_ino)==(after.st_dev,after.st_ino),'new claim inode replaced')
        need(P.read_json(P.child(self.queue.root,'attempt_state/'+self.shard['id']+'.json'),65536)[0]==self.token,'stale new generation refused')

    def sources(self):
        self.assert_current()
        if not hasattr(self,'_sources'):
            self._sources=self.queue.reader.read(self.shard['id']);self._source_map={s['source_id']:s for s in self._sources}
        return self._sources

    def assert_member(self,source):
        self.sources();need(self._source_map.get(source['source_id'])==source,'foreign source outside new claimed shard')

    def journal(self,value):
        self.assert_current()
        with (self.attempt_root/'journal.jsonl').open('ab')as f:f.write(P.canonical(value));f.flush();os.fsync(f.fileno())

    def finish(self,receipts):
        self.assert_current();sources={s['source_id']:s for s in self.sources()}
        need(len(receipts)==len(sources)and {r['source_id']for r in receipts}==set(sources),'incomplete/foreign new chunk refused')
        values=[self.queue._normalize_receipt(sources[r['source_id']],r)for r in receipts]
        value=dict(schema='redimnet2-bounded-chunk-commit/v1',shard_id=self.shard['id'],index_sha256=self.queue.index_sha,
                   contract_sha256=self.queue.contract_sha,shard_sha256=self.shard['sha256'],source_count=len(values),attempt_id=self.attempt_id,
                   sources=values,committed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_new(self.queue.chunk_receipt_path(self.shard['id']),value)
        self.assert_current();self.queue.write_completion(self.shard['id'],value)
        with new_lock(self.queue.root/'broker.lock'):
            self.assert_current();state=P.read_json(self.queue.root/'broker.json',1024**2)[0]
            need(state['leases'].get(self.shard['id'])==self.token,'new broker generation differs')
            state['leases'].pop(self.shard['id']);state['completed_chunks']+=1;replace_json(self.queue.root/'broker.json',state)
        return value

    def close(self):
        if not self.closed:self.closed=True;self.handle.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def inspect_bounded_progress(index_path,index_sha,queue_root,contract_sha):
    """只读新schema的单broker快照和已封账槽；不构造队列、不读侧车。"""
    reader=H.BoundedShardReader(index_path,index_sha);index=reader.index;root=P.physical(Path(queue_root))
    need(str(root)==index['new_queue_root']and H.is_sha(contract_sha),'explicit bounded queue/contract required')
    identity,identity_raw=P.read_json(root/'identity.json',65536)
    expected=dict(schema='redimnet2-bounded-work-queue/v1',index_sha256=index_sha,contract_sha256=contract_sha,
        old_contract_sha256=index['old_contract']['sha256'],source_count=index['source_count'],shard_count=len(index['shards']))
    need(identity==expected,'bounded progress queue identity differs')
    broker,broker_raw=P.read_json(root/'broker.json',1024**2)
    need(isinstance(broker,dict)and set(broker)=={'next','leases','completed_chunks'},'bad bounded broker fields')
    cursor=broker['next'];leases=broker['leases'];completed=broker['completed_chunks'];shards=index['shards'];by_id={s['id']:i for i,s in enumerate(shards)}
    need(type(cursor)is int and 0<=cursor<=len(shards)and isinstance(leases,dict)and type(completed)is int and completed==cursor-len(leases),'bad bounded broker cursor/count')
    need(all(sid in by_id and by_id[sid]<cursor and isinstance(v,dict)for sid,v in leases.items()),'bounded broker lease outside cursor')
    counts={k:0 for k in ('candidate','unknown','failed')};current=dict(counts);retained=dict(counts)
    total=0;slots=0;byte_count=0;slots_sha=hashlib.sha256()
    path=P.physical(root/'completion_map.bin');fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    try:
        before=os.fstat(fd);entry=path.stat()
        need(stat.S_ISREG(before.st_mode)and before.st_size==len(shards)*1024 and (before.st_dev,before.st_ino)==(entry.st_dev,entry.st_ino),'bounded completion file shape/identity differs')
        for ordinal in range(cursor):
            shard=shards[ordinal]
            if shard['id']in leases:continue
            data=os.pread(fd,1024,ordinal*1024);byte_count+=len(data)
            need(len(data)==1024 and data[:8]==b'RDMAP001','missing/torn bounded completion slot')
            size=struct.unpack('>I',data[8:12])[0];need(0<size<=980,'bad bounded completion payload size')
            raw=data[44:44+size]
            need(hashlib.sha256(raw).digest()==data[12:44]and data[44+size:]==bytes(980-size),'bounded completion checksum/padding differs')
            value=P.strict_json(raw)
            need(value.get('shard_id')==shard['id']and value.get('shard_sha256')==shard['sha256']and value.get('index_sha256')==index_sha and value.get('contract_sha256')==contract_sha and value.get('retained_producer_contract_sha256')==index['old_contract']['sha256'] and type(value.get('source_count'))is int and value['source_count']==shard['count'],'bounded completion producer/input differs')
            need(H.is_sha(value.get('commit_sha256'))and H.is_sha(value.get('source_ids_sha256')),'bounded completion receipt identity missing')
            for field in ('decision_counts','current_producer_decision_counts','retained_decision_counts'):
                d=value.get(field);need(isinstance(d,dict)and set(d)==set(counts)and all(type(n)is int and n>=0 for n in d.values()),'bounded decision counts malformed')
            c=value['current_producer_decision_counts'];r=value['retained_decision_counts'];all_counts=value['decision_counts']
            need(type(value.get('retained_source_count'))is int and type(value.get('current_producer_source_count'))is int and sum(r.values())==value['retained_source_count']and sum(c.values())==value['current_producer_source_count']and value['retained_source_count']+value['current_producer_source_count']==shard['count'],'bounded producer source counts differ')
            need(all(all_counts[k]==c[k]+r[k]for k in counts)and sum(all_counts.values())==shard['count'],'bounded producer decision accounting differs')
            for k in counts:counts[k]+=all_counts[k];current[k]+=c[k];retained[k]+=r[k]
            total+=shard['count'];slots+=1;slots_sha.update(data)
        after=os.fstat(fd);entry=path.stat()
        need(before.st_size==after.st_size and (before.st_dev,before.st_ino)==(after.st_dev,after.st_ino)==(entry.st_dev,entry.st_ino),'bounded completion file replaced/resized')
    finally:os.close(fd)
    need(slots==completed,'bounded completed slot count differs')
    need(index['previous_retained_count']+index['parent_preserved_source_count']+index['source_count']==index['global_source_count'],'bounded global scope accounting differs')
    global_done=index['previous_retained_count']+index['parent_preserved_source_count']+total
    return dict(schema='redimnet2-bounded-readonly-progress/v1',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        index_sha256=index_sha,contract_sha256=contract_sha,identity_sha256=P.byte_sha(identity_raw),broker_snapshot_sha256=P.byte_sha(broker_raw),
        broker_snapshot=broker,source_count=index['source_count'],shard_count=len(shards),committed_source_lower_bound=total,
        verified_completed_chunks=slots,completion_map_bytes_read=byte_count,completed_slots_sha256=slots_sha.hexdigest(),
        decision_counts=counts,current_producer_source_count=sum(current.values()),current_producer_decision_counts=current,
        previous_producer_retained_in_this_queue=sum(retained.values()),previous_producer_decision_counts=retained,
        global_source_count=index['global_source_count'],previous_retained_count=index['previous_retained_count'],
        parent_preserved_source_count=index['parent_preserved_source_count'],global_committed_source_lower_bound=global_done,
        unclaimed_JSONL_read_count=0,claimed_parent_JSONL_read_count=len(index['lease_partitions']),
        partial_JSONL_read_count=sum(s['storage']=='local_partial'for s in index['shards']),
        old_partial_commit_markers_read=index['retained_partial_source_count'],sidecars_reopened=0,
        chunk_receipt_payloads_reopened=0,database_created=False,queue_writes=0,locks_acquired=0,
        GPU_executed=False,human_gold_verified=False,full_sidecar_verification=False,
        current_producer_count_is_not_current_attempt_model_call_count=True,
        meaning='One broker snapshot; active leases excluded. Counts publication ownership, not actual new ASR calls, full sidecar bytes, human quality, or GPU resource return.')


def verify_bounded_terminal_metadata(index_path,index_sha,queue_root,contract_sha,*,terminal_pin=None,task_id=None,parent_lock_fd=None):
    """仅确认新queue全部固定槽已封账；不能当作全量侧车、人审或GPU归还验收。

    外部调用要求新鲜的实际平台终态并独占现有父锁；GPU父内部终态汇总则传其
    实际持有的parent_lock_fd，不在仍持锁的父内重新open竞争自己的锁。
    """
    root=P.physical(Path(queue_root));reader=H.BoundedShardReader(index_path,index_sha)
    if parent_lock_fd is None:
        need(terminal_pin is not None and isinstance(task_id,str),'actual terminal observation required outside parent')
        observed=P.terminal_observation(Path(terminal_pin['path']),terminal_pin['sha256'],task_id,root)
        lock_context=P.old_parent_lock(root)
        mode='external_terminal_and_existing_parent_lock'
    else:
        need(type(parent_lock_fd)is int and parent_lock_fd>2,'actual internal parent FD required')
        current=os.fstat(parent_lock_fd);path=P.physical(root/'.parent.lock');entry=path.stat()
        need(stat.S_ISREG(current.st_mode)and(current.st_dev,current.st_ino)==(entry.st_dev,entry.st_ino),'internal parent lock FD identity differs')
        if reader.index['purpose']=='production':
            info=Path('/proc/self/fdinfo')/str(parent_lock_fd)
            need(info.is_file()and any('FLOCK'in l and 'WRITE'in l for l in info.read_text().splitlines()if l.startswith('lock:')),'internal parent lock ownership not visible')
        observed=None;lock_context=contextlib.nullcontext();mode='internal_parent_owned_fd'
    with lock_context:
        first=inspect_bounded_progress(index_path,index_sha,root,contract_sha)
        need(first['verified_completed_chunks']==first['shard_count']and not first['broker_snapshot']['leases']and first['committed_source_lower_bound']==first['source_count'],'bounded queue is not completely committed')
        second=inspect_bounded_progress(index_path,index_sha,root,contract_sha)
        need(first['broker_snapshot_sha256']==second['broker_snapshot_sha256']and first['completed_slots_sha256']==second['completed_slots_sha256'],'bounded terminal metadata changed while read')
        if parent_lock_fd is not None:
            current=os.fstat(parent_lock_fd);entry=path.stat()
            need((current.st_dev,current.st_ino)==(entry.st_dev,entry.st_ino),'internal parent lock replaced during terminal read')
    return dict(schema='redimnet2-bounded-terminal-metadata/v1',metadata_complete=True,verification_mode=mode,
        terminal_observation=observed,progress=second,full_sidecar_verification=False,human_gold_verified=False,
        GPU_resource_return_verified_by_this_tool=False,formal_runtime_admitted=False,
        meaning='All selected queue slots committed and stable; this is not full sidecar/media/quality verification or CUDA resource return.')
