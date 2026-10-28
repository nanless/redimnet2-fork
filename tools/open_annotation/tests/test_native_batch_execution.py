"""显式原生批次的合同、真实调用包装和恢复CPU反例；没有真实模型/GPU。"""
from __future__ import annotations
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import contracts as C
from freeze_contract import freeze,make_native_formal_contract
from qwen_backend import QwenBackend,NativeBatchRefused,native_token_rows
from test_maintenance_entries import external_contract
from test_asr_compute_and_timing import Model,fake_torch


def profile(size):
    return dict(policy='fixed_cohort_until_durable/v1',max_requests=size,
                ordering='frames_then_row_id_within_source_group/v1')


def contract(size=4):
    value=external_contract();value['purpose']='cpu_fixture'
    value['runtime'].update(execution_policy=C.NATIVE_BATCH_POLICY,asr_batching=profile(size),
                            asr_compute={'dtype':'float32','attention':'sdpa'},max_new_tokens=512)
    return value


class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    @property
    def ndim(self):return self.value.ndim
    @property
    def shape(self):return self.value.shape
    def __getitem__(self,key):return Tensor(self.value[key])
    def detach(self):return self
    def cpu(self):return self
    def tolist(self):return self.value.tolist()


class NativeExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()

    @contextlib.contextmanager
    def backend(self,size=4,mode='normal'):
        wrappers=[];loads=[]
        def load(path,**options):
            loads.append(options);model=Model(dtype=options['dtype']);model.training=True
            def evaluate():model.training=False
            model.eval=evaluate;model.generation_config=types.SimpleNamespace(eos_token_id=2,pad_token_id=None)
            def generate(**kwargs):
                b=kwargs['input_ids'].shape[0]
                return Tensor([list(kwargs['input_ids'].tolist()[i])+[10+i,2,2]for i in range(b)])
            model.generate=generate
            processor=types.SimpleNamespace(tokenizer=types.SimpleNamespace(padding_side='right'),
                batch_decode=lambda values,**kwargs:['tok'+str(v[0])for v in values.tolist()])
            wrapper=types.SimpleNamespace(model=model,processor=processor,timestamp_token_id=151705,
                timestamp_segment_time=80,get_supported_languages=lambda:['English'])
            def transcribe(**kwargs):
                b=len(kwargs['audio']);inputs=Tensor([[0,9]if i==0 else[8,9]for i in range(b)])
                mask=Tensor([[0,1]if i==0 else[1,1]for i in range(b)])
                if mode=='wrong_width':inputs=Tensor([[8,9]]);mask=Tensor([[1,1]])
                result=model.generate(input_ids=inputs,attention_mask=mask,max_new_tokens=512)
                if mode=='double':model.generate(input_ids=inputs,attention_mask=mask,max_new_tokens=512)
                if mode=='missing_answer':b-=1
                return [types.SimpleNamespace(text='tok'+str(10+i),language='English')for i in range(b)]
            wrapper.transcribe=transcribe;wrappers.append(wrapper);return wrapper
        torch=fake_torch();torch.float32='float32';torch.bfloat16='bfloat16';torch.set_num_threads=lambda n:None
        torch.backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()),cudnn=types.SimpleNamespace())
        torch.cuda=types.SimpleNamespace(device_count=lambda:1,is_bf16_supported=lambda:True,synchronize=lambda n:None,
            max_memory_allocated=lambda n:100,max_memory_reserved=lambda n:200,OutOfMemoryError=MemoryError)
        torch.is_inference_mode_enabled=lambda:True;torch.is_grad_enabled=lambda:False;torch.inference_mode=contextlib.nullcontext
        qwen=types.SimpleNamespace(Qwen3ASRModel=types.SimpleNamespace(from_pretrained=load),
                                  Qwen3ForcedAligner=types.SimpleNamespace(from_pretrained=load))
        with patch.dict(sys.modules,{'torch':torch,'qwen_asr':qwen}),patch('qwen_backend.verify_models',return_value={}),\
             patch('qwen_backend.verify_model_stats'),patch.object(QwenBackend,'verify_loaded_modules',return_value={}):
            backend=QwenBackend(contract(size));backend.compute_observation={'synthetic_not_GPU_evidence':True}
            backend.prepare_native_group(self.root/'fixed-group','a'*64,'b'*64)
            yield backend,wrappers,loads

    def rows(self,n=4):
        rows=[]
        for i in range(n):
            wave=np.zeros(160+i,dtype=np.float32);source={'source_id':'src'+str(i),'dataset':'synthetic'}
            window={'window_id':'w0','input_pcm_sha256':hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),
                    'model_input_frames':len(wave),'model_sample_rate':16000}
            rows.append(dict(row_id=source['source_id']+':w0',audio=wave,context={'source':source},window=window))
        return rows

    def test_silver_and_implicit_old_policy_still_refuse_batching(self):
        c=contract();c['purpose']='silver_annotation'
        with self.assertRaisesRegex(RuntimeError,'pilot-only'):C.validate_execution_policy(c)
        c=contract();c['runtime']['execution_policy']='single_window_independent'
        with self.assertRaisesRegex(RuntimeError,'explicit native'):C.validate_execution_policy(c)
        for size in (True,0,2,16):
            c=contract(size)
            with self.subTest(size=size),self.assertRaises(RuntimeError):C.validate_execution_policy(c)

    def test_freeze_native_1_4_8_never_generates_silver(self):
        base=self.root/'base.json';base.write_bytes(C.canonical_bytes(external_contract()))
        original=base.read_bytes()
        for size in (1,4,8):
            receipt=freeze(base,C.sha_file(base),self.root/('batch'+str(size)),'native',
                           asr_compute={'dtype':'float32','attention':'sdpa'},asr_batching=profile(size))
            self.assertEqual(set(receipt['contracts']),{'pilot_validation'});self.assertTrue(receipt['native_ASR_batch_pilot_only'])
            c=C.read_contract(receipt['contracts']['pilot_validation']['path'])
            self.assertEqual(c['runtime']['max_batch'],1);self.assertEqual(C.asr_batch_profile(c),profile(size))
            self.assertIn('native_transformers_batch.py',c['engine_source_files'])
        self.assertEqual(base.read_bytes(),original)

    def test_formal_factory_is_explicit_default_disabled_data_without_math_changes(self):
        base=self.root/'base.json';base.write_bytes(C.canonical_bytes(external_contract()))
        frozen=freeze(base,C.sha_file(base),self.root/'pilot-code','candidate',asr_batching=profile(4))
        pin=frozen['contracts']['pilot_validation'];out=self.root/'formal-control'/'contract.json'
        receipt=make_native_formal_contract(pin['path'],pin['sha256'],out,'explicit-formal-data')
        self.assertFalse(receipt['default_enabled']);self.assertFalse(receipt['formal_runtime_admitted'])
        self.assertFalse(receipt['GPU_executed']);self.assertFalse(receipt['training_labels_released'])
        pilot=json.loads(Path(pin['path']).read_text());formal=C.read_contract(out,receipt['new_contract']['sha256'])
        self.assertNotIn('native_formal',pilot);self.assertEqual(formal['native_formal'],C.NATIVE_FORMAL_DECLARATION)
        self.assertEqual({k:v for k,v in pilot.items()if k not in {'purpose','version'}},
                         {k:v for k,v in formal.items()if k not in {'purpose','version','native_formal'}})
        with self.assertRaisesRegex(RuntimeError,'fresh'):make_native_formal_contract(pin['path'],pin['sha256'],out,'another')

    def test_formal_declaration_cannot_enable_old_or_pilot_policies_or_numeric_false(self):
        for purpose,policy,decl in [('pilot_validation',C.NATIVE_BATCH_POLICY,C.NATIVE_FORMAL_DECLARATION),
                ('silver_annotation',C.NATIVE_BATCH_POLICY,dict(C.NATIVE_FORMAL_DECLARATION,default_enabled=0)),
                ('silver_annotation','single_window_independent',C.NATIVE_FORMAL_DECLARATION)]:
            c=contract();c['purpose']=purpose;c['runtime']['execution_policy']=policy;c['native_formal']=copy.deepcopy(decl)
            with self.subTest(purpose=purpose,policy=policy,decl=decl),self.assertRaises(RuntimeError):C.validate_execution_policy(c)

    def test_real_generate_wrapper_records_width_order_padding_and_zero_call_recovery(self):
        with self.backend()as(b,wrappers,loads):
            self.assertEqual(loads[0]['max_inference_batch_size'],4)
            self.assertEqual(loads[1]['dtype'],'float32');self.assertEqual(loads[1]['attn_implementation'],'eager')
            first=b.transcribe(self.rows());self.assertEqual(b.stats['asr_call_rows'],{'4':1})
            self.assertEqual(b.stats['asr_rows'],4);self.assertEqual(b.stats['asr_generated_tokens'],8)
            self.assertEqual(b.stats['asr_returned_token_slots'],12)
            for i,row in enumerate(self.rows()):
                value=first[row['row_id']];native=value['native']
                self.assertEqual(native['actual_generate_row_index'],i);self.assertEqual(native['actual_generate_rows'],4)
                self.assertEqual(native['effective_generated_token_ids'],[10+i,2]);self.assertEqual(native['post_eos_token_ids'],[2])
                self.assertFalse(value['durable_native_batch']['cache_hit'])
            second=b.transcribe(self.rows()[1:]);self.assertEqual(b.stats['asr_rows'],4)
            self.assertTrue(all(v['durable_native_batch']['cache_hit']for v in second.values()))
            self.assertEqual(b.stats['native_ASR_stage_cache_hits'],3)
            self.assertEqual(b.metrics()['asr_call_rows'],{'4':1})
            self.assertFalse(list(self.root.rglob('*.sqlite*')))

    def test_split_or_wrong_generate_width_is_fatal_and_preserves_attempt(self):
        for mode in ('wrong_width','double','missing_answer'):
            with self.subTest(mode=mode):
                # 每次使用不同组目录，上一失败的意图和输出证据不删。
                with self.backend(mode=mode)as(b,_,__):
                    b.prepare_native_group(self.root/mode,'a'*64,'b'*64)
                    with self.assertRaises(Exception)as seen:b.transcribe(self.rows())
                    self.assertTrue(hasattr(seen.exception,'native_batch_evidence'))
                    self.assertTrue(list((self.root/mode/'attempts').glob('*.json')))
                    self.assertFalse(list((self.root/mode/'batches').glob('*.json')))

    def test_legacy_single_window_guard_is_not_bypassed_by_new_fields(self):
        b=QwenBackend.__new__(QwenBackend);b.contract=external_contract()
        b.stats={'asr_call_rows':{},'aligner_call_rows':{}}
        with self.assertRaises(RuntimeError):b._observe_call('asr',self.rows())
        b.contract=contract()
        with self.assertRaises(RuntimeError):b._observe_call('aligner',self.rows())

    def test_worker_whole_group_store_survives_partial_sidecar_publication_without_ASR_cache(self):
        import worker
        rows=self.rows();sources=[r['context']['source']for r in rows]
        for i,s in enumerate(sources):s['audio_path']='/synthetic/'+str(i)
        by_id={r['context']['source']['source_id']:r for r in rows};published={};records={}
        def identity(source):
            row=by_id[source['source_id']]
            return dict(decode_status='decoded_complete',frames=len(row['audio']),sample_rate=16000,channels=1,
                        duration_seconds=len(row['audio'])/16000,stat_after={'source':source['source_id']})
        def publish(source,record,claim):
            records[source['source_id']]=copy.deepcopy(record)
            receipt=dict(source_id=source['source_id'],decision='unknown');published[source['source_id']]=receipt;return receipt
        store=types.SimpleNamespace(root=self.root/'output-tree',lookup=lambda s:published.get(s['source_id']),publish=publish)
        claim=types.SimpleNamespace(shard={'id':'fixed-shard'},attempt_id='synthetic-attempt',
            assert_member=lambda s:None,journal=lambda value:None)
        args=types.SimpleNamespace(contract_sha='a'*64,worker_id='CPU')
        class NoASRCache:
            def get(self,stage,inputs):
                if stage=='asr':raise AssertionError('new policy must not read per-window ASR cache')
                return None
            def put(self,stage,inputs,value):
                if stage=='asr':raise AssertionError('new policy must not publish per-window ASR cache')
                return value
        def counters():return dict(assigned_source_rows=0,sidecars_committed=0,sidecar_cache_hits=0,
            retained_previous_producer_hits=0,retained_previous_decision_counts=dict(candidate=0,unknown=0,failed=0),
            decision_counts=dict(candidate=0,unknown=0,failed=0),stage_cache_hits=dict(asr=0,aligner=0),
            model_windows=0,decoded_audio_seconds=0,error_source_count=0,errors=[],pipeline_seconds={})
        with self.backend()as(b,_,__),patch.object(worker,'inspect_source',side_effect=lambda s:(identity(s),[])),\
             patch.object(worker,'reuse_vad',return_value={'errors':[]}),\
             patch.object(worker,'plan_windows',side_effect=lambda ident,s,*args:[copy.deepcopy(by_id[s['source_id']]['window'])]),\
             patch.object(worker,'read_window',side_effect=lambda s,*args:by_id[s['source_id']]['audio']),\
             patch.object(worker,'units_from_windows',return_value=([],[])),\
             patch.object(worker,'stat_identity',side_effect=lambda p:{'source':'src'+p.rsplit('/',1)[1]}),\
             patch.object(worker,'build_record',side_effect=lambda s,ident,desc,vad,windows,units,execution,errors:dict(source=s,windows=windows,errors=errors)),\
             patch.object(b,'align',side_effect=lambda values:{r['row_id']:dict(status='not_applicable',reason='synthetic_only')for r in values}):
            descriptor={'provenance':{'models':C.model_provenance(b.contract),'code_sha256':'b'*64}}
            first=counters();worker.process_group(sources,claim,store,NoASRCache(),b,b.contract,args,descriptor,first)
            plan=b._native_store.root/'plan.json';before=plan.read_bytes();self.assertEqual(b.stats['asr_rows'],4)
            # 模拟整批已持久、其中只有一个完整侧车留下；来源组仍然是原四条。
            keep=published['src0'];published.clear();published['src0']=keep
            second=counters();worker.process_group(sources,claim,store,NoASRCache(),b,b.contract,args,descriptor,second)
            self.assertEqual(b._native_store.root/'plan.json',plan);self.assertEqual(plan.read_bytes(),before)
            self.assertEqual(b.stats['asr_rows'],4);self.assertEqual(second['sidecar_cache_hits'],1)
            self.assertEqual(second['stage_cache_hits']['asr'],3);self.assertEqual(second['sidecars_committed'],3)
            self.assertTrue(all(records[s]['windows'][0]['asr_cache_hit']for s in ('src1','src2','src3')))


class NativeTokenGeometryTests(unittest.TestCase):
    def test_early_eos_and_truncated_row_are_distinguished(self):
        rows=native_token_rows([[0,9],[8,9]],[[0,1],[1,1]],[[3,2,2],[4,5,6]],['a','b'],[2],2,3)
        self.assertTrue(rows[0]['eos_observed']);self.assertEqual(rows[0]['effective_generated_token_count'],2)
        self.assertFalse(rows[1]['eos_observed']);self.assertEqual(rows[1]['effective_generated_token_count'],3)
        self.assertEqual(rows[0]['input_nonpad_tokens'],1)

    def test_mask_holes_right_padding_unknown_suffix_and_missing_rows_refused(self):
        bad=[([[1,0]],[[3,2]],['a']),([[0,0]],[[3,2]],['a']),([[1,1]],[[2,7]],['a']),([[1,1]],[],[])]
        for masks,tokens,text in bad:
            with self.subTest(masks=masks,tokens=tokens),self.assertRaises(NativeBatchRefused):
                native_token_rows([[8,9]],masks,tokens,text,[2],2,3)


if __name__=='__main__':unittest.main()
