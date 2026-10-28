"""新队列固定元数据与完成槽；临时小文件，无来源/侧车/模型。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import bounded_metadata_progress as M


def raw(value):return(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()


class MetadataProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.queue=self.root/'queue';self.queue.mkdir();self.contract='c'*64
        self.index={'schema':'annotation-bounded-source-index/v1','shards':[{'id':'partial-a','sha256':'d'*64,'count':2,'storage':'local_partial'},
            {'id':'full-b','sha256':'e'*64,'count':3,'storage':'parent_unclaimed','parent_ordinal':8}],
            'source_count':5,'shard_count':2,'parent_cursor':8,'old_contract':{'sha256':'b'*64},'new_queue_root':str(self.queue),
            'previous_retained_count':20,'parent_preserved_source_count':10,'global_source_count':35}
        self.publish_inputs()
        self.slots=[self.slot(self.index['shards'][0]),self.slot(self.index['shards'][1])];self.map.write_bytes(b''.join(self.encode(s)for s in self.slots));self.broker={'next':2,'leases':{},'completed_chunks':2};self.save_broker()

    @property
    def map(self):return self.queue/'completion_map.bin'

    def put(self,name,value):
        p=self.root/name;p.write_bytes(raw(value));return{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

    def publish_inputs(self):
        self.ip=self.put('index.json',self.index);self.rp=self.put('report.json',{'status':'terminal_thin_plan_not_gpu_admission','new_index':self.ip,'old_producer_modified':False,'database_created':False,'training_labels_released':False,'exact_remaining_source_count':5,'global_source_count':35,'previous_retained_count':20,'retained_complete_source_count':8,'retained_partial_source_count':2})
        (self.queue/'identity.json').write_bytes(raw({'schema':'redimnet2-bounded-work-queue/v1','index_sha256':self.ip['sha256'],'contract_sha256':self.contract,'old_contract_sha256':'b'*64,'source_count':5,'shard_count':2}))

    def slot(self,s):
        count=s['count'];counts={'candidate':count,'unknown':0,'failed':0};return{'shard_id':s['id'],'shard_sha256':s['sha256'],'source_count':count,'source_ids_sha256':'f'*64,'commit_sha256':'a'*64,'index_sha256':self.ip['sha256'],'contract_sha256':self.contract,'retained_producer_contract_sha256':'b'*64,'decision_counts':counts,'current_producer_source_count':count,'retained_source_count':0,'current_producer_decision_counts':counts,'retained_decision_counts':{'candidate':0,'unknown':0,'failed':0}}

    def encode(self,value):
        b=raw(value);self.assertLessEqual(len(b),980);return b'RDMAP001'+struct.pack('>I',len(b))+hashlib.sha256(b).digest()+b+bytes(980-len(b))

    def save_broker(self):(self.queue/'broker.json').write_bytes(raw(self.broker))
    def inspect(self):return M.inspect(self.ip,self.rp,self.queue,self.contract)

    def test_actual_metadata_only_no_source_constructor_or_file_writes(self):
        allowed={self.ip['path'],self.rp['path'],str(self.queue/'identity.json'),str(self.queue/'broker.json')};original=M.read_small
        def bounded(path,maximum):
            self.assertIn(str(path),allowed);return original(path,maximum)
        with patch.object(M,'read_small',side_effect=bounded),patch.object(Path,'mkdir',side_effect=AssertionError('no queue creation')):
            result=self.inspect()
        self.assertEqual(result['committed_source_lower_bound'],5);self.assertEqual(result['global_committed_source_lower_bound'],35);self.assertEqual(result['current_producer_source_count'],5)
        self.assertEqual(result['completed_by_storage']['local_partial']['sources'],2);self.assertEqual(result['completed_by_storage']['parent_unclaimed']['sources'],3)
        self.assertEqual(result['source_JSONL_reads'],0);self.assertFalse(result['runtime_admitted']);self.assertFalse(result['source_closed_scope_revalidated'])

    def test_claimed_slot_is_excluded_even_if_torn(self):
        self.broker={'next':2,'leases':{'full-b':{'owner':'fixture'}},'completed_chunks':1};self.save_broker();self.map.write_bytes(self.encode(self.slots[0])+bytes(1024));result=self.inspect();self.assertEqual(result['committed_source_lower_bound'],2);self.assertEqual(result['completion_map_bytes_read'],1024)

    def test_torn_checksum_and_extra_slot_fields_refused(self):
        original=self.encode(self.slots[0])+self.encode(self.slots[1]);self.map.write_bytes(b'x'+original[1:])
        with self.assertRaisesRegex(ValueError,'torn'):self.inspect()
        self.map.write_bytes(original[:20]+b'x'+original[21:])
        with self.assertRaisesRegex(ValueError,'checksum'):self.inspect()
        changed=dict(self.slots[0],extra=True);self.map.write_bytes(self.encode(changed)+self.encode(self.slots[1]))
        with self.assertRaisesRegex(ValueError,'fields'):self.inspect()

    def test_reordered_slots_and_foreign_producer_refused(self):
        self.map.write_bytes(self.encode(self.slots[1])+self.encode(self.slots[0]))
        with self.assertRaisesRegex(ValueError,'ordinal/producer'):self.inspect()
        wrong=dict(self.slots[0],contract_sha256='0'*64);self.map.write_bytes(self.encode(wrong)+self.encode(self.slots[1]))
        with self.assertRaisesRegex(ValueError,'ordinal/producer'):self.inspect()

    def test_counts_bool_extra_state_and_missing_completion_refused(self):
        for change in ({'current_producer_source_count':True},{'decision_counts':{'candidate':2,'unknown':0,'failed':0,'extra':0}},{'retained_source_count':1}):
            self.map.write_bytes(self.encode(dict(self.slots[0],**change))+self.encode(self.slots[1]))
            with self.subTest(change=change),self.assertRaises(ValueError):self.inspect()
        self.map.write_bytes(self.encode(self.slots[0])+bytes(1024))
        with self.assertRaisesRegex(ValueError,'torn'):self.inspect()

    def test_new_and_retained_ownership_remain_distinct(self):
        v=copy.deepcopy(self.slots[0]);v.update(current_producer_source_count=1,retained_source_count=1,current_producer_decision_counts={'candidate':1,'unknown':0,'failed':0},retained_decision_counts={'candidate':1,'unknown':0,'failed':0});self.map.write_bytes(self.encode(v)+self.encode(self.slots[1]));d=self.inspect();self.assertEqual(d['current_producer_source_count'],4);self.assertEqual(d['previous_producer_retained_in_this_queue'],1);self.assertEqual(d['preserved_before_new_queue'],30)

    def test_reordered_metadata_and_changed_index_pin_refused(self):
        self.index['shards'].reverse();self.publish_inputs()
        with self.assertRaisesRegex(ValueError,'ordering'):self.inspect()
        Path(self.ip['path']).write_bytes(raw(self.index)+b' ')
        with self.assertRaisesRegex(ValueError,'pin SHA'):self.inspect()

    def test_negative_balanced_report_counts_refused(self):
        report=json.loads(Path(self.rp['path']).read_text());report.update(retained_complete_source_count=-8,retained_partial_source_count=18);self.rp=self.put('report.json',report)
        with self.assertRaisesRegex(ValueError,'report accounting'):self.inspect()

    def test_boolean_unclaimed_ordinal_cannot_masquerade_as_zero(self):
        self.index['parent_cursor']=0;self.index['shards'][1]['parent_ordinal']=False;self.publish_inputs();self.slots=[self.slot(s)for s in self.index['shards']];self.map.write_bytes(b''.join(self.encode(s)for s in self.slots))
        with self.assertRaisesRegex(ValueError,'ordinals'):self.inspect()


if __name__=='__main__':unittest.main()
