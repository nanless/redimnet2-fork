"""只用合成PCM和假原生结果核整批事务，不调用模型、CUDA或数据库。"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import native_transformers_batch as M


class NativeFileBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.rows = []
        for i in range(8):
            wave = np.full(160+i,i/10,dtype=np.float32)
            source = {'source_id':'src'+str(i),'relative_path':str(i)+'.wav','roles':['train_ptn']}
            self.rows.append(dict(row_id=source['source_id']+':w',source_id=source['source_id'],window_id='w',
                source_record_sha256=M.digest(source),input_pcm_sha256=hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),
                model_input_frames=len(wave),model_sample_rate=16000,audio=wave,context={'source':source}))
        self.calls = []

    def store(self,width=4,**changes):
        args=dict(synthetic_fixture=True,contract_sha256='b'*64,model_sha256='c'*64,max_batch_rows=width)
        args.update(changes)
        return M.NativeTransformersBatchStore(self.root/'store','a'*64,**args)

    def infer(self,rows):
        self.calls.append([r['row_id']for r in rows])
        return {row['row_id']:dict(status='completed',raw_text='完整字',normalized_text='完整字',language='Chinese',
            native=dict(actual_generate_rows=len(rows),actual_generate_row_index=i,generated_token_ids=[11,2],
                        first_eos_index=1,eos_observed=True,max_new_tokens=512))for i,row in enumerate(rows)}

    def test_exact_batch4_plan_two_atomic_packets_no_database(self):
        with patch('sqlite3.connect',side_effect=AssertionError('database forbidden')):
            store=self.store();out=store.run_batch(self.rows,infer_batch=self.infer)
        self.assertEqual([len(v)for v in self.calls],[4,4]);self.assertEqual(set(out),{r['row_id']for r in self.rows})
        self.assertEqual(store.metrics['new_model_rows'],8);self.assertFalse(store.metrics['database_created'])
        for key,value in out.items():
            ref=value['durable_native_batch'];packet,sha=M.read_json(Path(ref['packet_path']))
            self.assertEqual(sha,ref['packet_sha256']);self.assertEqual(len(packet['content']['outputs']),4)
            self.assertIn(key,packet['content']['outputs'])
        self.assertFalse(list(self.root.rglob('*.sqlite*')))

    def test_width8_and_new_code_width1_baseline(self):
        for width in (8,1):
            with self.subTest(width=width):
                store=M.NativeTransformersBatchStore(self.root/str(width),'a'*64,synthetic_fixture=True,contract_sha256='b'*64,model_sha256='c'*64,max_batch_rows=width)
                self.calls=[];store.run_batch(self.rows,infer_batch=self.infer)
                self.assertEqual([len(v)for v in self.calls],[width]*(8//width))

    def test_committed_subset_resume_zero_model_calls_no_audio_needed(self):
        store=self.store();before=store.run_batch(self.rows,infer_batch=self.infer)
        subset=[{k:v for k,v in self.rows[i].items()if k!='audio'}for i in (7,1)]
        with patch.object(self,'infer',side_effect=AssertionError('published batch must not rerun')):
            out=self.store().run_batch(subset,infer_batch=self.infer)
        for row in subset:
            key=row['row_id'];expected=copy.deepcopy(before[key]);expected['durable_native_batch']['cache_hit']=True
            self.assertEqual(out[key],expected)

    def test_uncommitted_original_cohort_cannot_drop_or_reorder(self):
        store=self.store()
        def fail(rows):raise RuntimeError('before any result was committed')
        with self.assertRaisesRegex(RuntimeError,'before any'):store.run_batch(self.rows,infer_batch=fail)
        for rows in (self.rows[1:4],list(reversed(self.rows[:4]))):
            with self.subTest(ids=[r['row_id']for r in rows]),self.assertRaisesRegex(RuntimeError,'all original rows'):
                self.store().run_batch(rows,infer_batch=self.infer)
        self.assertEqual(self.calls,[])

    def test_exception_after_durable_packet_preserves_evidence_and_reuses_whole_batch(self):
        store=self.store()
        def stop(_):raise RuntimeError('synthetic interruption after whole commit')
        with self.assertRaisesRegex(RuntimeError,'after whole'):store.run_batch(self.rows,infer_batch=self.infer,after_durable_commit=stop)
        self.assertEqual(len(self.calls),1)
        evidence=json.loads(next((self.root/'store/attempts').glob('*.json')).read_text())
        self.assertTrue(evidence['complete_batch_committed_before_error']);self.assertEqual(len(evidence['native_outputs']),4)
        self.calls=[];out=self.store().run_batch(self.rows,infer_batch=self.infer)
        self.assertEqual(self.calls,[[r['row_id']for r in self.rows[4:]]]);self.assertEqual(len(out),8)
        self.assertTrue(out[self.rows[0]['row_id']]['durable_native_batch']['cache_hit'])
        self.assertFalse(out[self.rows[4]['row_id']]['durable_native_batch']['cache_hit'])

    def test_partial_native_result_never_becomes_per_position_cache(self):
        store=self.store()
        def partial(rows):return {rows[0]['row_id']:self.infer(rows)[rows[0]['row_id']]}
        with self.assertRaisesRegex(RuntimeError,'whole native cohort'):store.run_batch(self.rows,infer_batch=partial)
        self.assertFalse((self.root/'store/batches').exists())
        evidence=json.loads(next((self.root/'store/attempts').glob('*.json')).read_text())
        self.assertEqual(len(evidence['native_outputs']),1);self.assertFalse(evidence['complete_batch_committed_before_error'])
        self.assertIn('coverage differs',evidence['traceback'])

    def test_foreign_or_duplicate_native_position_rejected(self):
        for mutation in ('width','position','foreign'):
            folder=self.root/mutation;store=M.NativeTransformersBatchStore(folder,'a'*64,synthetic_fixture=True,contract_sha256='b'*64,model_sha256='c'*64,max_batch_rows=4)
            def bad(rows):
                out=self.infer(rows)
                if mutation=='width':out[rows[0]['row_id']]['native']['actual_generate_rows']=1
                elif mutation=='position':out[rows[1]['row_id']]['native']['actual_generate_row_index']=0
                else:out['foreign']=out.pop(rows[0]['row_id'])
                return out
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):store.run_batch(self.rows[:4],infer_batch=bad)
            self.assertFalse((folder/'batches').exists())

    def test_actual_pcm_mismatch_rejected_before_callback(self):
        self.rows[0]['audio'][0]+=1
        with self.assertRaisesRegex(RuntimeError,'PCM SHA'):self.store().run_batch(self.rows,infer_batch=self.infer)
        self.assertEqual(self.calls,[])

    def test_source_record_mismatch_and_window_limits_rejected(self):
        self.rows[0]['context']['source']['roles']=['test']
        with self.assertRaisesRegex(RuntimeError,'complete actual source'):self.store().run_batch(self.rows,infer_batch=self.infer)
        self.rows[0]['source_record_sha256']=M.digest(self.rows[0]['context']['source']);self.rows[0]['model_input_frames']=480001
        with self.assertRaisesRegex(RuntimeError,'30 seconds'):self.store().run_batch(self.rows,infer_batch=self.infer)

    def test_changed_model_contract_profile_or_producer_cannot_reuse(self):
        self.store().run_batch(self.rows,infer_batch=self.infer)
        for changes in ({'model_sha256':'f'*64},{'contract_sha256':'f'*64},{'max_batch_rows':8},{'synthetic_fixture':False}):
            with self.subTest(changes=changes),self.assertRaisesRegex(RuntimeError,'another producer'):self.store(**changes)

    def test_unseen_request_cannot_make_new_plan_in_same_store(self):
        store=self.store();store.run_batch(self.rows[:4],infer_batch=self.infer)
        with self.assertRaisesRegex(RuntimeError,'outside fixed plan'):store.run_batch(self.rows[4:],infer_batch=self.infer)

    def test_modified_packet_or_fixed_plan_refused_without_rerun(self):
        store=self.store();store.run_batch(self.rows,infer_batch=self.infer);packet=next((self.root/'store/batches').glob('*.json'))
        saved=packet.read_bytes();x=json.loads(saved);next(iter(x['content']['outputs'].values()))['raw_text']='changed';packet.write_text(json.dumps(x))
        with self.assertRaisesRegex(RuntimeError,'packet digest'):self.store().run_batch(self.rows,infer_batch=self.infer)
        packet.write_bytes(saved);p=self.root/'store/plan.json';x=json.loads(p.read_text());x['cohorts'].pop();p.write_text(json.dumps(x))
        with self.assertRaisesRegex(RuntimeError,'incomplete or changed'):self.store().run_batch(self.rows,infer_batch=self.infer)

    def test_pending_cohort_requires_actual_pcm_cache_does_not(self):
        rows=[{k:v for k,v in r.items()if k!='audio'}for r in self.rows[:4]]
        with self.assertRaisesRegex(RuntimeError,'every original actual PCM'):self.store().run_batch(rows,infer_batch=self.infer)
        self.assertEqual(self.calls,[])

    def test_callback_exception_preserves_captured_partial_native_evidence(self):
        def broken(rows):
            exc=RuntimeError('second native generate was refused')
            exc.native_batch_evidence={'actual_generate_count':2,'expected_row_ids':[r['row_id']for r in rows],
                'captures':[self.infer(rows)],'partial_native_matrix':[[11,2,2]],
                'padding_evidence':{'pad_token_id':2,'post_eos_token_ids':[2],'input_attention_mask':[0,1,1]}}
            raise exc
        with self.assertRaisesRegex(RuntimeError,'second native'):self.store().run_batch(self.rows,infer_batch=broken)
        saved=json.loads(next((self.root/'store/attempts').glob('*.json')).read_text())
        self.assertEqual(saved['native_callback_evidence']['actual_generate_count'],2)
        self.assertEqual(len(saved['native_callback_evidence']['captures'][0]),4)
        self.assertEqual(saved['native_callback_evidence']['expected_row_ids'],[r['row_id']for r in self.rows[:4]])
        self.assertEqual(saved['native_callback_evidence']['partial_native_matrix'],[[11,2,2]])
        self.assertEqual(saved['native_callback_evidence']['padding_evidence']['post_eos_token_ids'],[2])

    def test_symlink_and_unknown_tree_preserved(self):
        target=self.root/'elsewhere';target.mkdir();link=self.root/'link';link.symlink_to(target,target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError,'physical'):M.NativeTransformersBatchStore(link,'a'*64,synthetic_fixture=True,contract_sha256='b'*64,model_sha256='c'*64,max_batch_rows=4)
        unknown=self.root/'store';unknown.mkdir();(unknown/'evidence').write_text('keep')
        with self.assertRaisesRegex(RuntimeError,'unknown existing'):self.store()
        self.assertEqual((unknown/'evidence').read_text(),'keep')


if __name__=='__main__':unittest.main()
