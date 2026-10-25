"""Synthetic fixed-slot queues only; no WorkQueue, model, GPU or real task."""
import copy
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('readonly_progress', Path(__file__).resolve().parents[1] / 'readonly_progress.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def raw_json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def slot(value):
    raw = raw_json(value)
    return M.SLOT_MAGIC + struct.pack('>I', len(raw)) + hashlib.sha256(raw).digest() + raw + bytes(980-len(raw))


class ReadonlyProgressTests(unittest.TestCase):
    def fixture(self, root, count=4):
        queue = root / 'queue'; queue.mkdir()
        shards = [{'id': 's%05d' % i, 'path': 'unused/%d.jsonl' % i,
                   'sha256': hashlib.sha256(str(i).encode()).hexdigest(), 'count': 2} for i in range(count)]
        index = {'schema': 'redimnet2-shard-index/v1', 'source_count': count*2, 'shards': shards}
        ip = root / 'index.json'; ip.write_bytes(raw_json(index)); pin = hashlib.sha256(ip.read_bytes()).hexdigest()
        contract = 'c' * 64
        identity = {'schema':'redimnet2-work-queue/v1','index_sha256':pin,'contract_sha256':contract,
                    'source_count':count*2,'shard_count':count}
        (queue/'identity.json').write_bytes(raw_json(identity))
        broker = {'next':3,'leases':{'s00001':{'worker_id':'synthetic'}},'completed_chunks':2}
        (queue/'broker.json').write_bytes(raw_json(broker))
        with (queue/'completion_map.bin').open('wb') as stream: stream.truncate(count*1024)
        values = []
        for i in (0,2):
            value = {'shard_id':shards[i]['id'],'shard_sha256':shards[i]['sha256'],'source_count':2,
                'source_ids_sha256':'a'*64,'commit_sha256':'b'*64,'index_sha256':pin,'contract_sha256':contract,
                'decision_counts':{'candidate':1,'unknown':1,'failed':0}}
            values.append(value)
            with (queue/'completion_map.bin').open('r+b') as stream:stream.seek(i*1024);stream.write(slot(value))
        return ip,pin,queue,contract,values

    def inspect(self, fixture):return M.inspect_progress(*fixture[:4])

    def replace_slot(self,fixture,value):
        with (fixture[2]/'completion_map.bin').open('r+b') as stream:stream.write(slot(value))

    def test_large_sparse_queue_reads_only_snapshot_completed_slots_and_never_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve(); f=self.fixture(root,45975)
            before={str(p.relative_to(root)):(p.stat().st_size,p.stat().st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in root.rglob('*') if p.is_file()}
            pread=os.pread
            with patch.object(M.os,'pread',wraps=pread) as read:
                result=self.inspect(f)
            self.assertEqual(read.call_count,2)
            self.assertEqual([call.args[2]for call in read.call_args_list],[0,2048])
            self.assertEqual(result['completion_map_bytes_read'],2048)
            self.assertEqual(result['committed_source_lower_bound'],4)
            self.assertEqual(result['decision_counts'],{'candidate':2,'unknown':2,'failed':0})
            self.assertFalse(result['full_sidecar_verification']);self.assertIsNone(result['ETA'])
            after={str(p.relative_to(root)):(p.stat().st_size,p.stat().st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before,after)

    def test_corrupt_missing_short_and_bad_length_slots_refused(self):
        for kind in ('magic','checksum','missing','length','short'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory()as temp:
                f=self.fixture(Path(temp).resolve());p=f[2]/'completion_map.bin'
                with p.open('r+b')as stream:
                    if kind=='short':stream.truncate(1024)
                    elif kind=='missing':stream.write(bytes(1024))
                    elif kind=='length':stream.seek(8);stream.write(struct.pack('>I',981))
                    else:stream.seek(0 if kind=='magic' else 44);stream.write(b'!')
                with self.assertRaises(ValueError):self.inspect(f)

    def test_pinned_slot_fields_and_decision_denominator_refused(self):
        changes=[('shard_id','foreign'),('shard_sha256','f'*64),('index_sha256','f'*64),
                 ('contract_sha256','f'*64),('source_count',True),('source_count',3),
                 ('commit_sha256','bad'),('source_ids_sha256','bad')]
        for key,value in changes:
            with self.subTest(key=key,value=value),tempfile.TemporaryDirectory()as temp:
                f=self.fixture(Path(temp).resolve());item=copy.deepcopy(f[4][0]);item[key]=value;self.replace_slot(f,item)
                with self.assertRaises(ValueError):self.inspect(f)
        for values in ({'candidate':True,'unknown':1,'failed':0},{'candidate':-1,'unknown':3,'failed':0},
                       {'candidate':1,'unknown':0,'failed':0},{'candidate':1,'unknown':1,'failed':0,'extra':0}):
            with self.subTest(values=values),tempfile.TemporaryDirectory()as temp:
                f=self.fixture(Path(temp).resolve());item=copy.deepcopy(f[4][0]);item['decision_counts']=values;self.replace_slot(f,item)
                with self.assertRaises(ValueError):self.inspect(f)

    def test_broker_cursor_count_and_leases_rejected(self):
        for update in ({'completed_chunks':1},{'next':True},{'next':5},{'leases':{'s00003':{}}},{'leases':{'foreign':{}}}):
            with self.subTest(update=update),tempfile.TemporaryDirectory()as temp:
                f=self.fixture(Path(temp).resolve());p=f[2]/'broker.json';broker=json.loads(p.read_bytes());broker.update(update);p.write_bytes(raw_json(broker))
                with self.assertRaises(ValueError):self.inspect(f)

    def test_broker_is_opened_once_even_if_newer_atomic_snapshot_appears(self):
        with tempfile.TemporaryDirectory()as temp:
            f=self.fixture(Path(temp).resolve());original=M.read_small;calls=[]
            def read(path,maximum):
                value,raw=original(path,maximum)
                if Path(path).name=='broker.json':
                    calls.append(path)
                    (f[2]/'broker.json').write_bytes(raw_json({'next':4,'leases':{},'completed_chunks':4}))
                return value,raw
            with patch.object(M,'read_small',side_effect=read):result=self.inspect(f)
            self.assertEqual(len(calls),1);self.assertEqual(result['verified_completed_chunks'],2)

    def test_atomic_broker_replace_during_open_fd_read_keeps_valid_old_snapshot(self):
        with tempfile.TemporaryDirectory()as temp:
            f=self.fixture(Path(temp).resolve());fdopen=M.os.fdopen;opened=[]
            @contextlib.contextmanager
            def intercept(fd,*args,**kwargs):
                with fdopen(fd,*args,**kwargs)as stream:
                    opened.append(fd)
                    if len(opened)==3:  # index, identity, then the single broker FD
                        replacement=f[2]/'new-broker.tmp'
                        replacement.write_bytes(raw_json({'next':4,'leases':{},'completed_chunks':4}))
                        os.replace(replacement,f[2]/'broker.json')
                    yield stream
            with patch.object(M.os,'fdopen',side_effect=intercept):result=self.inspect(f)
            self.assertEqual(result['verified_completed_chunks'],2)
            self.assertEqual(result['broker_snapshot']['next'],3)

    def test_duplicate_JSON_keys_index_SHA_and_queue_identity_rejected(self):
        for kind in ('duplicate','nan','pin','identity'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory()as temp:
                f=self.fixture(Path(temp).resolve())
                if kind=='duplicate':(f[2]/'broker.json').write_bytes(b'{"next":3,"next":3,"leases":{},"completed_chunks":3}')
                elif kind=='nan':(f[2]/'broker.json').write_bytes(b'{"next":NaN,"leases":{},"completed_chunks":0}')
                elif kind=='pin':f=(f[0],'0'*64,*f[2:])
                else:(f[2]/'identity.json').write_bytes(b'{}')
                with self.assertRaises(ValueError):self.inspect(f)

    def test_output_refuses_queue_existing_and_dangling_symlink_before_read(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp).resolve();f=self.fixture(root);existing=root/'preserve.json';existing.write_text('preserve')
            link=root/'dangling';target=root/'absent';link.symlink_to(target)
            for output in (f[2]/'report.json',existing,link):
                with self.subTest(output=output),patch.object(M,'inspect_progress',side_effect=AssertionError('no input reads')) as read:
                    with self.assertRaises(ValueError):M.main(['--index',str(f[0]),'--index-sha',f[1],'--queue-root',str(f[2]),'--contract-sha',f[3],'--output',str(output)])
                    read.assert_not_called()
            self.assertEqual(existing.read_text(),'preserve');self.assertFalse(target.exists())

    def test_cli_publishes_only_explicit_new_report(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp).resolve();f=self.fixture(root);out=root/'new.json'
            code=M.main(['--index',str(f[0]),'--index-sha',f[1],'--queue-root',str(f[2]),'--contract-sha',f[3],'--output',str(out)])
            self.assertEqual(code,0);self.assertEqual(json.loads(out.read_bytes())['committed_source_lower_bound'],4)
            self.assertFalse(any(p.name.endswith('.lock')for p in root.rglob('*')))


if __name__=='__main__':unittest.main()
