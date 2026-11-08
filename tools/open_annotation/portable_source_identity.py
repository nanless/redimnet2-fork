"""显式跨节点内容证明：保存新旧stat，以实际编码SHA证明原音未变。

只为新后继协议生成独立账本；不修改旧侧车、合同、队列或冻结源码。
默认严格stat读取规则由调用方保留，本模块不会自动降级或跳过字段。
"""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
import stat
import time
from urllib.parse import quote

if __package__:
    from .readonly_progress import read_small
else:
    from readonly_progress import read_small

SPEC_SCHEMA='annotation-portable-source-spec/v1'
ENTRY_SCHEMA='annotation-portable-source-evidence/v1'
MODE='portable_encoded_sha256/v1'
VERIFIED='encoded_content_verified'
FAILED='historical_decode_failure_preserved_not_content_verified'
STAT_KEYS={'device','inode','bytes','mtime_ns','ctime_ns'}


def need(value,message):
    if not value:raise ValueError(message)


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def is_sha(value):return isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None


def physical(value):
    p=Path(value)
    need(p.is_absolute()and '..'not in p.parts and all(not x.is_symlink()for x in (p,*p.parents)),
         'physical absolute path without parent traversal required')
    return p


def absolute(value):
    need(isinstance(value,str)and Path(value).is_absolute()and '..'not in Path(value).parts,
         'explicit safe absolute identity path required')
    return Path(value)


def validate_portable_spec(value):
    need(isinstance(value,dict)and value.get('schema')==SPEC_SCHEMA and
         value.get('purpose')in {'production','cpu_fixture'}and value.get('source_identity_mode')==MODE,
         'explicit portable source schema/purpose/mode required')
    need(value.get('allowed_difference')==['device']and value.get('encoded_hash_algorithm')=='sha256',
         'only device may differ; actual encoded SHA256 is mandatory')
    need(type(value.get('model_API_calls_allowed'))is int and value['model_API_calls_allowed']==0 and
         value.get('release_training_labels')is False,'portable source proof cannot call API or release labels')
    for name in ('old_contract','full_index'):
        pin=value.get(name);need(isinstance(pin,dict)and set(pin)=={'path','sha256'}and is_sha(pin['sha256']),
                                'explicit original pin required: '+name)
        absolute(pin['path'])
    for name in ('old_queue_root','old_sidecar_tree'):absolute(value[name])
    alias=value.get('source_alias')
    need(isinstance(alias,dict)and set(alias)=={'alias_path','target_path'},'one explicit source mount alias required')
    a=absolute(alias['alias_path']);t=absolute(alias['target_path'])
    need(a!=t and a not in t.parents and t not in a.parents,'alias and target must be independent exact roots')
    if value['purpose']=='production':
        need(str(a)=='/root/group-shared'and str(t)=='/shared/group-shared','unrecognized production mount alias')
    return value


def load_portable_spec(path,expected_sha,*,contract_pin=None,index_pin=None,queue=None,tree=None):
    need(is_sha(expected_sha),'portable spec SHA required')
    value,raw=read_small(physical(path),1024**2)
    need(hashlib.sha256(raw).hexdigest()==expected_sha,'portable spec bytes differ')
    validate_portable_spec(value)
    for name,expected in (('old_contract',contract_pin),('full_index',index_pin),
                          ('old_queue_root',str(queue)if queue is not None else None),
                          ('old_sidecar_tree',str(tree)if tree is not None else None)):
        if expected is not None:need(value[name]==expected,'portable spec differs from original input: '+name)
    return value


def resolve_media_path(spec,source_path):
    """仅解析明确的一层共享别名；不改变source中的原路径原文。"""
    validate_portable_spec(spec)
    alias=Path(spec['source_alias']['alias_path']);target=physical(spec['source_alias']['target_path'])
    physical(alias.parent)
    need(alias.is_symlink()and target.is_dir()and alias.resolve(strict=True)==target,
         'actual exact source alias target differs')
    source=absolute(source_path)
    if alias in source.parents:return physical(target/source.relative_to(alias))
    if spec['purpose']=='production':need(target in source.parents,'source outside explicitly bound shared mount')
    return physical(source)


def stat_identity(value):
    s=value if isinstance(value,os.stat_result)else Path(value).stat()
    return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}


def valid_stat(value):
    return isinstance(value,dict)and set(value)==STAT_KEYS and all(type(v)is int and v>=0 for v in value.values())


class SidecarInventory:
    """一次目录扫描建新SQLite清单；大量missing来源只查清单，不stat PFS。"""
    def __init__(self,path,tree,spec,spec_sha,domains,progress=None):
        self.path=physical(path);self.tree=physical(tree);self.spec=spec;self.spec_sha=spec_sha
        need(not self.path.exists()and self.path.parent.is_dir(),'fresh external inventory database required')
        fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600);os.close(fd)
        self.db=sqlite3.connect('file:'+quote(str(self.path),safe='/')+'?mode=rw',uri=True)
        self.db.execute('PRAGMA journal_mode=DELETE');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA temp_store=FILE');self.db.execute('PRAGMA cache_size=-8192')
        self.db.execute('CREATE TABLE paths(relative TEXT PRIMARY KEY,sidecar INTEGER NOT NULL,commit_marker INTEGER NOT NULL,seen INTEGER NOT NULL DEFAULT 0,source_id TEXT,kind TEXT)')
        self.db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.domains=set(domains);self.counts={'directories_scanned':0,'sidecar_files':0,'commit_files':0,
            'known_identity_files':0,'known_stage_cache_roots_skipped':0,'original_sources_checked':0,
            'missing_source_lookups_without_PFS_stat':0,'matched_completed_sources':0,'matched_partial_sources':0,
            'matched_unclaimed_sources':0};self.pending=0;self.finished=False
        self.progress=progress;self.last_progress=time.monotonic();self.phase='inventory_scan'

    def emit_progress(self,event=None,force=False):
        now=time.monotonic()
        if callable(self.progress)and (force or now-self.last_progress>=30):
            self.progress(event or self.phase+'_progress',dict(self.counts));self.last_progress=now

    def scan(self):
        self.emit_progress('inventory_scan_started',True)
        def walk(directory):
            self.counts['directories_scanned']+=1
            self.emit_progress()
            with os.scandir(directory)as entries:
                for entry in entries:
                    need(not entry.is_symlink(),'sidecar inventory symlink preserved and refused: '+entry.path)
                    relative=str(Path(entry.path).relative_to(self.tree))
                    if entry.is_dir(follow_symlinks=False):
                        if relative=='.stage_cache':self.counts['known_stage_cache_roots_skipped']+=1;continue
                        if Path(entry.path).parent==self.tree:
                            need(entry.name in self.domains,'unknown root directory in old sidecar tree: '+relative)
                        walk(Path(entry.path));continue
                    need(entry.is_file(follow_symlinks=False),'nonregular sidecar inventory entry: '+relative)
                    if relative=='identity.json':self.counts['known_identity_files']+=1;continue
                    if relative.endswith('.annotation.json.commit.json'):
                        key=relative[:-len('.commit.json')];sidecar=0;commit=1;self.counts['commit_files']+=1
                    elif relative.endswith('.annotation.json'):
                        key=relative;sidecar=1;commit=0;self.counts['sidecar_files']+=1
                    else:raise ValueError('unknown residual file preserved in old sidecar tree: '+relative)
                    self.db.execute('INSERT INTO paths(relative,sidecar,commit_marker)VALUES(?,?,?) ON CONFLICT(relative)DO UPDATE SET sidecar=sidecar+excluded.sidecar,commit_marker=commit_marker+excluded.commit_marker',
                                    (key,sidecar,commit));self.pending+=1
                    self.emit_progress()
                    if self.pending>=5000:self.db.commit();self.pending=0
        walk(self.tree);self.db.commit()
        bad=self.db.execute('SELECT relative FROM paths WHERE sidecar!=1 OR commit_marker!=1 LIMIT 1').fetchone()
        need(bad is None,'orphan or duplicate sidecar/commit inventory pair preserved: '+str(bad))
        need(self.counts['known_identity_files']==1,'old namespace identity missing from inventory')
        self.emit_progress('inventory_scan_finished',True)

    def select(self,source,kind):
        need(kind in {'completed','partial','unclaimed'},'explicit old selection kind required')
        relative=source['sidecar_relpath'];path=Path(relative)
        need(isinstance(relative,str)and not path.is_absolute()and path.parts and
             not any(p in {'.','..'}for p in path.parts)and '\\'not in relative and '\x00'not in relative and
             relative==source['dataset']+'/'+source['relative_path']+'.annotation.json',
             'unsafe or changed source sidecar mirror mapping')
        self.counts['original_sources_checked']+=1
        row=self.db.execute('SELECT sidecar,commit_marker,seen FROM paths WHERE relative=?',(relative,)).fetchone()
        if row is None:
            need(kind!='completed','completed source sidecar pair missing from directory inventory')
            self.counts['missing_source_lookups_without_PFS_stat']+=1;self.emit_progress();return False
        need(row==(1,1,0),'duplicate original source/path or changed inventory pair')
        self.db.execute('UPDATE paths SET seen=1,source_id=?,kind=? WHERE relative=?',
                        (source['source_id'],kind,source['sidecar_relpath']))
        self.pending+=1
        if self.pending>=5000:self.db.commit();self.pending=0
        self.counts['matched_'+kind+'_sources']+=1;self.emit_progress();return True

    def finish(self,full_count):
        need(self.counts['original_sources_checked']==full_count,'inventory did not check every original source record')
        extra=self.db.execute('SELECT relative FROM paths WHERE seen!=1 LIMIT 1').fetchone()
        need(extra is None,'old sidecar path absent from original index; preserved and refused: '+str(extra))
        summary={'schema':'annotation-portable-sidecar-inventory/v1','portable_spec_sha256':self.spec_sha,
            'old_contract':self.spec['old_contract'],'full_index':self.spec['full_index'],
            'old_sidecar_tree':str(self.tree),'all_inventory_paths_matched_original_sources':True,
            'source_audio_read_by_inventory':False,'completed_sidecar_content_read_by_inventory':False,
            'stage_cache_traversed':False,**self.counts}
        for key,value in summary.items():self.db.execute('INSERT INTO metadata VALUES(?,?)',(key,json.dumps(value,ensure_ascii=False,sort_keys=True)))
        self.db.commit();self.finished=True;return summary

    def close(self):
        self.db.commit();self.db.close()


def _hash_encoded_fd(fd):
    h=hashlib.sha256()
    with os.fdopen(fd,'rb',closefd=False)as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def prove_source_identity(spec,spec_sha,source,record,contract_sha,sidecar_sha,commit_sha):
    """读取实际编码原音；成功前不写账本，失败原件保持原样。"""
    validate_portable_spec(spec)
    need(is_sha(spec_sha)and is_sha(contract_sha)and is_sha(sidecar_sha)and is_sha(commit_sha),'bound source proof SHA missing')
    need(spec['old_contract']['sha256']==contract_sha and record['source']==source and
         record['contract']['sha256']==contract_sha,'portable proof producer/source differs')
    identity=record['source_identity']
    entry={'schema':ENTRY_SCHEMA,'source_identity_mode':MODE,'purpose':spec['purpose'],
        'portable_spec_sha256':spec_sha,'source_id':source['source_id'],'source_record_sha256':digest(source),
        'sidecar_sha256':sidecar_sha,'commit_sha256':commit_sha,'old_contract_sha256':contract_sha,
        'source_audio_path':source['audio_path'],'resolved_media_path':None,
        'old_encoded_file_sha256':identity.get('file_sha256'),'actual_encoded_file_sha256':None,
        'old_stat_before':identity.get('stat_before'),'old_stat_after':identity.get('stat_after'),
        'actual_stat_before':None,'actual_stat_after':None,'stat_differences':[],
        'old_source_identity_sha256':digest(identity),'old_pcm_sha256':identity.get('pcm_sha256'),
        'old_pcm_hash_policy':identity.get('pcm_hash_policy'),'PCM_redecoded':False,
        'encoded_bytes_read':0,'encoded_content_verified':False,
        'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observer_hostname':socket.gethostname(),
        'source_or_record_rewritten':False,'training_or_evaluation_eligibility_released':False}
    if identity['decode_status']!='decoded_complete':
        need(record['decision']['state']=='failed','unverified decoded source cannot be retained as a successful candidate')
        entry['status']=FAILED
    else:
        old=identity.get('stat_after')
        need(valid_stat(old)and identity.get('stat_before')==old and old['bytes']==identity.get('bytes')and
             is_sha(identity.get('file_sha256')),'old full stat and encoded source SHA required')
        media=resolve_media_path(spec,source['audio_path'])
        fd=os.open(media,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
        try:
            raw_before=os.fstat(fd);need(stat.S_ISREG(raw_before.st_mode),'encoded source must be a regular file')
            before=stat_identity(raw_before)
            need(stat_identity(media)==before,'source path/FD identity differs before read')
            differences=sorted(k for k in STAT_KEYS if old[k]!=before[k])
            need(set(differences)<={'device'},'portable identity refuses non-device stat drift: '+
                 json.dumps({'source_id':source['source_id'],'differences':differences,'old':old,'actual':before},sort_keys=True))
            actual_sha=_hash_encoded_fd(fd)
            after=stat_identity(os.fstat(fd))
            need(before==after==stat_identity(media),'encoded source full stat changed during SHA read')
            need(resolve_media_path(spec,source['audio_path'])==media,'source alias changed during SHA read')
            need(actual_sha==identity['file_sha256'],'actual encoded source SHA differs from old published identity')
        finally:os.close(fd)
        entry.update(status=VERIFIED,resolved_media_path=str(media),actual_encoded_file_sha256=actual_sha,
            actual_stat_before=before,actual_stat_after=after,stat_differences=differences,
            encoded_bytes_read=before['bytes'],encoded_content_verified=True)
    validate_ledger_entry(entry,source,record,contract_sha,spec,spec_sha=spec_sha)
    return entry


def validate_ledger_entry(entry,source,record,old_contract_sha,portable_spec=None,*,spec_sha=None):
    """只核已保存账本与原记录绑定；不读取或声称重新核过当前原音。"""
    need(isinstance(entry,dict)and entry.get('schema')==ENTRY_SCHEMA and entry.get('source_identity_mode')==MODE,
         'wrong portable source evidence schema/mode')
    need(is_sha(old_contract_sha),'portable evidence old contract SHA invalid')
    observed=datetime.datetime.fromisoformat(entry['observed_utc'])
    need(observed.tzinfo is not None and isinstance(entry.get('observer_hostname'),str)and bool(entry['observer_hostname'].strip()),
         'portable content observer/timezone missing')
    need(entry.get('purpose')in {'production','cpu_fixture'}and is_sha(entry.get('portable_spec_sha256')),
         'portable evidence purpose/spec SHA missing')
    if spec_sha is not None:need(entry['portable_spec_sha256']==spec_sha,'portable evidence spec SHA differs')
    need(entry.get('source_id')==source['source_id']and entry.get('source_record_sha256')==digest(source)and
         record.get('source')==source and record['contract']['sha256']==old_contract_sha and
         entry.get('old_contract_sha256')==old_contract_sha and entry.get('source_audio_path')==source['audio_path']and
         is_sha(entry.get('sidecar_sha256'))and is_sha(entry.get('commit_sha256')),
         'portable evidence source/producer/sidecar binding differs')
    need(entry['sidecar_sha256']==digest(record),'portable evidence old sidecar bytes differ')
    need(entry.get('source_or_record_rewritten')is False and entry.get('training_or_evaluation_eligibility_released')is False and
         record['decision']['usable_for_training']is False and record['decision']['usable_for_evaluation']is False,
         'portable source proof cannot rewrite records or release labels')
    identity=record['source_identity']
    need(entry.get('old_source_identity_sha256')==digest(identity)and entry.get('old_pcm_sha256')==identity.get('pcm_sha256')and
         entry.get('old_pcm_hash_policy')==identity.get('pcm_hash_policy')and entry.get('PCM_redecoded')is False,
         'portable evidence old PCM/clock identity binding differs')
    need(entry.get('old_stat_before')==identity.get('stat_before')and entry.get('old_stat_after')==identity.get('stat_after')and
         entry.get('old_encoded_file_sha256')==identity.get('file_sha256'),'portable evidence old identity altered')
    if portable_spec is not None:
        validate_portable_spec(portable_spec)
        need(portable_spec['old_contract']['sha256']==old_contract_sha and entry['purpose']==portable_spec['purpose'],
             'portable evidence bound spec producer/purpose differs')
    if identity['decode_status']!='decoded_complete':
        need(entry.get('status')==FAILED and record['decision']['state']=='failed'and
             all(entry.get(k)is None for k in ('resolved_media_path','actual_encoded_file_sha256','actual_stat_before','actual_stat_after'))and
             entry.get('stat_differences')==[]and type(entry.get('encoded_bytes_read'))is int and entry['encoded_bytes_read']==0 and
             entry.get('encoded_content_verified')is False,'historical decode failure cannot claim fresh content verification')
        return {'content_verified':False,'historical_failure_preserved':True}
    old=identity['stat_after'];before=entry.get('actual_stat_before');after=entry.get('actual_stat_after')
    need(entry.get('status')==VERIFIED and valid_stat(old)and identity['stat_before']==old and valid_stat(before)and
         before==after and is_sha(identity.get('file_sha256'))and entry.get('actual_encoded_file_sha256')==identity['file_sha256'],
         'portable content hash/full stable stat missing or contradictory')
    need(entry.get('encoded_content_verified')is True and type(entry.get('encoded_bytes_read'))is int and
         entry['encoded_bytes_read']==before['bytes'],'portable evidence encoded read count differs')
    differences=sorted(k for k in STAT_KEYS if old[k]!=before[k])
    need(set(differences)<={'device'}and entry.get('stat_differences')==differences and old['bytes']==identity['bytes'],
         'portable evidence hides a non-device difference')
    resolved=absolute(entry.get('resolved_media_path'))
    if portable_spec is not None:
        alias=Path(portable_spec['source_alias']['alias_path']);target=Path(portable_spec['source_alias']['target_path']);original=Path(source['audio_path'])
        expected=target/original.relative_to(alias)if alias in original.parents else original
        need(resolved==expected,'portable evidence resolved media path differs from explicit alias')
    return {'content_verified':True,'historical_failure_preserved':False}
