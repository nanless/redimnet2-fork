"""Synthetic CPU engineering tests, never evidence of spoken-word quality."""
import importlib.util as _deps
import unittest
if any(_deps.find_spec(p) is None for p in ['torch', 'soundfile']):
    raise unittest.SkipTest('optional CPU test dependencies unavailable: '+', '.join(['torch', 'soundfile']))
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import contextmanager
import numpy as np
import soundfile as sf
import torch
from torch import nn
from word_ft import schema
from word_ft import train as T

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(2,4);self.bn=nn.BatchNorm1d(4);self.drop=nn.Dropout(.2)
    def forward(self,x):
        return self.drop(self.bn(self.linear(torch.stack((x.mean(-1),x.square().mean(-1)),1))))
class TinyHead(nn.Module):
    def __init__(self,n):super().__init__();self.linear=nn.Linear(4,n);self.margin=.3
    def forward(self,x,y):
        z=self.linear(x);return z,nn.functional.cross_entropy(z,y)
def tiny_load(*args):return TinyBackbone(),{'embed_dim':4}
def tiny_head(dim,n,seed,args):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed);return TinyHead(n)
def config():
    return {'seed':10,'head_seed':4,'items_per_rank':2,'total_steps':4,'checkpoint_every':2,
        'optimizer':{'lr':.001,'momentum':.9,'nesterov':True,'weight_decay':.0001},
        'head':{'margin_type':'C','dynamic_margin':False},'official_sha256':T.OFFICIAL_SHA,
        'bn_policy':'frozen_statistics_trainable_affine','amp':False,'decoder':{},'allow_candidates':False,'sampling':'individual_units'}

class FakeDDP(nn.Module):
    def __init__(self,model):super().__init__();self.model=model;self.inside=False;self.calls=[]
    @contextmanager
    def no_sync(self):
        self.inside=True
        try:yield
        finally:self.inside=False
    def forward(self,*args):self.calls.append(self.inside);return self.model(*args)

def ddp_worker(rank,port,data,out,stop,resume):
    os.environ.update(WORLD_SIZE='2',RANK=str(rank),LOCAL_RANK=str(rank),
                      MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),CUDA_VISIBLE_DEVICES='')
    with patch.object(T,'load_official',tiny_load),patch.object(T,'build_head',tiny_head),\
         patch.object(T,'decoder_identity',return_value={'fixture':'synthetic CPU decoder'}):
        T.run(config(),data,'unused',out,resume=resume,stop_after=stop,cpu_threads=1)

def run_ddp_fixture(data,out,stop=None,resume=None):
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    torch.multiprocessing.spawn(ddp_worker,args=(port,str(data),str(out),stop,str(resume) if resume else None),nprocs=2,join=True)

class TrainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        records=[]
        for i,n in enumerate([2300,4999,7101,1903,9091]):
            audio=self.root/f'{i}.wav';sf.write(audio,np.sin(np.arange(n)*(.02+i*.003)).astype('float32'),16000,subtype='FLOAT')
            records.append(schema.make_unit(audio,f's{i%3}','你'*(i+1),split='train',
                provenance={'fixture':'synthetic waveform; not real text alignment'}))
        self.dataset=schema.write_dataset(self.root/'units.json',records,{'split':'train','synthetic_fixture':True})
        self.data=self.root/'units.json'
    def run_tiny(self,out,resume=None,stop=None,c=None):
        with patch.object(T,'load_official',tiny_load),patch.object(T,'build_head',tiny_head),\
             patch.object(T,'decoder_identity',return_value={'fixture':'synthetic CPU decoder'}):
            return T.run(c or config(),self.data,'unused',out,resume=resume,stop_after=stop)
    def test_true_item_lengths_and_resume_bitwise(self):
        full=self.root/'full';split=self.root/'split'
        self.run_tiny(full);self.run_tiny(split,stop=2);self.run_tiny(split,resume=split/'train_step_2.pt')
        a=torch.load(full/'train_step_4.pt',weights_only=False);b=torch.load(split/'train_step_4.pt',weights_only=False)
        for name in ['state_dict','head']:
            for key in a[name]:torch.testing.assert_close(a[name][key],b[name][key],rtol=0,atol=0)
        for i,st in a['optimizer']['state'].items():
            for key,value in st.items():torch.testing.assert_close(value,b['optimizer']['state'][i][key],rtol=0,atol=0)
        self.assertTrue(torch.equal(a['rank_rngs'][0]['torch'],b['rank_rngs'][0]['torch']))
        self.assertEqual(a['rank_rngs'][0]['python'],b['rank_rngs'][0]['python'])
        self.assertTrue(np.array_equal(a['rank_rngs'][0]['numpy'][1],b['rank_rngs'][0]['numpy'][1]))
        self.assertEqual(a['exposure'],b['exposure']);self.assertEqual(a['initial_head_sha256'],b['initial_head_sha256'])
        aa=[json.loads(x) for x in (full/'exposure_rank0.jsonl').read_text().splitlines()]
        bb=[json.loads(x) for x in (split/'exposure_rank0.jsonl').read_text().splitlines()]
        for x,y in zip(aa,bb):
            for key in ['items','loss','item_losses','keys','exposure_global']:self.assertEqual(x[key],y[key])
        actual=[it['input_samples'] for row in aa for it in row['items']]
        self.assertGreater(len(set(actual)),1);self.assertEqual(a['exposure']['items'],8)
        self.assertTrue(torch.equal(a['state_dict']['bn.running_mean'],torch.zeros(4)))
        self.assertEqual(a['state_dict']['bn.num_batches_tracked'].item(),0)
        public=torch.load(full/'embedding_step_4.pt',weights_only=False)
        self.assertEqual(set(public),{'model_config','state_dict'})
    def make_families(self):
        records=[]
        for i in range(5):
            audio=self.root/f'family{i}.wav';n=5000+i*311
            sf.write(audio,np.sin(np.arange(n)*.031).astype('float32'),16000,subtype='FLOAT')
            for count,end in zip([1,2,3,6],[601,1501,2801,n]):
                r=schema.make_unit(audio,f's{i%3}','一二三四五六'[:count],end_sample=end,
                    evidence='reference_timed_segment',split='train',source_utterance_id=f'utterance{i}',
                    provenance={'fixture':'synthetic only; not provider real boundaries'})
                r['family_id']=f'family{i}';records.append(r)
        self.data=self.root/'families.json';schema.write_dataset(self.data,records,{'split':'train','synthetic_fixture':True})
        return records
    def family_config(self,mixed):
        c=config();c.update(sampling='matched_prefix_family',unit_counts=[1,2,3,6] if mixed else [6],
                           unit_probabilities=[1/6,1/6,1/6,.5] if mixed else [1.])
        return c
    def test_matched_family_m1_m2_order_anchor_head_and_real_lengths(self):
        self.make_families();a=self.run_tiny(self.root/'m1',c=self.family_config(False))
        b=self.run_tiny(self.root/'m2',c=self.family_config(True))
        self.assertEqual(a['initial_head_sha256'],b['initial_head_sha256'])
        aa=[json.loads(x) for x in (self.root/'m1/exposure_rank0.jsonl').read_text().splitlines()]
        bb=[json.loads(x) for x in (self.root/'m2/exposure_rank0.jsonl').read_text().splitlines()]
        for x,y in zip(aa,bb):
            for long,short in zip(x['items'],y['items']):
                self.assertEqual(long['n_units'],6)
                for key in ['family_id','audio_sha256','speaker_id','source_utterance_id','start_sample','sampler_global_position']:
                    self.assertEqual(long[key],short[key])
                self.assertEqual(short['unit_texts'],long['unit_texts'][:short['n_units']])
                self.assertLessEqual(short['end_sample'],long['end_sample'])
                self.assertEqual(short['n_units'],T.count_for_position(self.family_config(True),short['sampler_global_position']))
        self.assertNotEqual(a['exposure']['units'],b['exposure']['units'])
    def test_family_missing_duplicate_source_prefix_and_end_refused(self):
        records=self.make_families();T.build_family_index(records)
        for kind in ['missing','duplicate','speaker','anchor','text','end']:
            changed=copy.deepcopy(records)
            if kind=='missing':changed.pop(0)
            elif kind=='duplicate':changed.append(changed[0])
            elif kind=='speaker':changed[0]['speaker_id']='other'
            elif kind=='anchor':changed[0]['start_sample']=1
            elif kind=='text':changed[0]['unit_texts']=['不']
            else:changed[1]['end_sample']=changed[0]['end_sample']
            with self.subTest(kind=kind),self.assertRaises(ValueError):T.build_family_index(changed)
    def test_count_rng_does_not_consume_model_or_global_sampler_rng(self):
        T.seed_all(12);before=T.capture_rng()
        a=[T.count_for_position(self.family_config(True),i) for i in range(100)]
        b=[T.count_for_position(self.family_config(True),i) for i in range(100)]
        after=T.capture_rng();self.assertEqual(a,b)
        self.assertTrue(torch.equal(before['torch'],after['torch']))
        self.assertEqual(before['python'],after['python'])
        self.assertTrue(np.array_equal(before['numpy'][1],after['numpy'][1]))
    def test_optional_vad_is_intersection_only_and_bound_to_source(self):
        r=copy.deepcopy(self.dataset['records'][0]);self.assertIsNone(T.predicted_vad_seconds(r))
        r['vad']={'segments_source_samples':[[0,100],[200,400]],'source_sha256':r['audio_sha256']}
        self.assertEqual(T.predicted_vad_seconds(r),300/16000)
        r['start_sample']=250;self.assertEqual(T.predicted_vad_seconds(r),150/16000)
        r['vad']['source_sha256']='bad'
        with self.assertRaisesRegex(ValueError,'identity'):T.predicted_vad_seconds(r)
        r['vad']['source_sha256']=r['audio_sha256'];r['vad']['segments_source_samples']=[[0,300],[200,400]]
        with self.assertRaisesRegex(ValueError,'geometry'):T.predicted_vad_seconds(r)
    def test_real_two_rank_cpu_accumulation_resume(self):
        full=self.root/'ddp-full';split=self.root/'ddp-split'
        run_ddp_fixture(self.data,full)
        run_ddp_fixture(self.data,split,stop=2)
        run_ddp_fixture(self.data,split,resume=split/'train_step_2.pt')
        a=torch.load(full/'train_step_4.pt',weights_only=False);b=torch.load(split/'train_step_4.pt',weights_only=False)
        for name in ['state_dict','head']:
            for key in a[name]:torch.testing.assert_close(a[name][key],b[name][key],rtol=0,atol=0)
        for i,st in a['optimizer']['state'].items():
            for key,value in st.items():torch.testing.assert_close(value,b['optimizer']['state'][i][key],rtol=0,atol=0)
        self.assertEqual(a['exposure'],b['exposure']);self.assertEqual(a['exposure']['items'],16)
        for r in range(2):
            self.assertTrue(torch.equal(a['rank_rngs'][r]['torch'],b['rank_rngs'][r]['torch']))
            aa=[json.loads(x) for x in (full/f'exposure_rank{r}.jsonl').read_text().splitlines()]
            bb=[json.loads(x) for x in (split/f'exposure_rank{r}.jsonl').read_text().splitlines()]
            for x,y in zip(aa,bb):
                for key in ['items','loss','item_losses','keys','exposure_global']:self.assertEqual(x[key],y[key])
    def test_accumulation_is_equal_item_mean_and_no_sync_forward(self):
        T.seed_all(3);model=T.FineTuneModel(TinyBackbone(),TinyHead(3));model.backbone.drop=nn.Identity();model.train();T.freeze_bn(model)
        reference=copy.deepcopy(model);wrapper=FakeDDP(model)
        waves=[np.linspace(-.4,.4,n,dtype=np.float32) for n in [300,999,1731]];labels=[0,1,2]
        opt=torch.optim.SGD(model.parameters(),lr=.01);refopt=torch.optim.SGD(reference.parameters(),lr=.01)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        T.accumulate_step(wrapper,model,opt,scaler,waves,labels,torch.device('cpu'))
        refopt.zero_grad()
        loss=sum(reference(torch.from_numpy(w).unsqueeze(0),torch.tensor([y]))[2] for w,y in zip(waves,labels))/3
        loss.backward();refopt.step()
        for key,value in model.state_dict().items():torch.testing.assert_close(value,reference.state_dict()[key],rtol=0,atol=2e-9)
        self.assertEqual(wrapper.calls,[True,True,False])
    def test_no_tail_dropped_and_reproducible_order(self):
        stream=sum([T.step_indices(5,s,2,1,4) for s in range(5)],[])
        self.assertEqual(sorted(stream[:5]),list(range(5)));self.assertEqual(sorted(stream[5:]),list(range(5)))
        self.assertEqual(T.step_indices(5,4,2,1,4),stream[8:10])
    def test_default_refuses_candidates_but_explicit_silver_runs(self):
        records=copy.deepcopy(self.dataset['records'])
        for r in records:
            r.update(evidence='forced_alignment_candidate',boundary_review='pending',alignment={'synthetic':True})
            r['unit_id']=schema.digest(schema.unit_identity(r))
        self.data=self.root/'candidates.json';schema.write_dataset(self.data,records,{'split':'train'})
        with self.assertRaisesRegex(RuntimeError,'not a reference'):self.run_tiny(self.root/'reject')
        c=config();c['allow_candidates']=True;self.run_tiny(self.root/'silver',stop=1,c=c)
        identity=json.loads((self.root/'silver/identity_rank0_start_step_0.json').read_text())
        self.assertIn('silver',identity['candidate_policy'])
    def test_changed_data_config_or_source_refuses_resume(self):
        out=self.root/'split';self.run_tiny(out,stop=2)
        c=config();c['items_per_rank']=3
        with self.assertRaisesRegex(ValueError,'identity mismatch'):self.run_tiny(out,resume=out/'train_step_2.pt',c=c)
        with patch.object(T,'source_identity',return_value={'changed':'sha'}):
            with self.assertRaisesRegex(ValueError,'source identity'):self.run_tiny(out,resume=out/'train_step_2.pt')
    def test_stale_step_and_incomplete_evidence_refused(self):
        out=self.root/'full';self.run_tiny(out)
        with self.assertRaisesRegex(ValueError,'stale'):T.validate_resume_target(out,out/'train_step_2.pt',1,0)
        evidence=out/'exposure_rank0.jsonl';evidence.write_bytes(evidence.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError,'partial evidence'):T.validate_resume_target(out,out/'train_step_4.pt',1,0)
    def test_resume_writer_lock_and_under_lock_recheck(self):
        import fcntl
        out=self.root/'split';self.run_tiny(out,stop=2)
        with open(out/'.training.lock','r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):T.validate_resume_target(out,out/'train_step_2.pt',1,0)
            T.validate_resume_target(out,out/'train_step_2.pt',1,0,writer_lock_held=True)
        self.run_tiny(out,resume=out/'train_step_2.pt')
        with self.assertRaisesRegex(ValueError,'stale'):
            T.validate_resume_target(out,out/'train_step_2.pt',1,0,writer_lock_held=True)
    def test_resume_rechecks_earlier_item_rows_not_just_last_counter(self):
        out=self.root/'split';self.run_tiny(out,stop=2)
        evidence=out/'exposure_rank0.jsonl'
        rows=[json.loads(x) for x in evidence.read_text().splitlines()]
        rows[0]['items'][0]['input_samples']+=1
        evidence.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with self.assertRaisesRegex(ValueError,'position/length'):
            T.validate_resume_target(out,out/'train_step_2.pt',1,0)
    def test_decode_exception_retained_no_success_subset(self):
        with patch.object(T,'_read_record',side_effect=RuntimeError('exact decode failure')):
            with self.assertRaisesRegex(RuntimeError,'exact decode failure'):self.run_tiny(self.root/'failed')
        out=self.root/'failed'
        self.assertIn('Traceback',(out/'failure_rank0_step0.txt').read_text())
        self.assertFalse(list(out.glob('train_step_*.pt')))
        self.assertEqual((out/'exposure_rank0.jsonl').read_text(),'')
    def test_source_interval_preserved_and_resampling_supported(self):
        r=self.dataset['records'][0];w=T._read_record(r)
        self.assertEqual(len(w),r['end_sample']-r['start_sample'])
        a=self.root/'8k.wav';sf.write(a,np.arange(777,dtype=np.float32)/1000,8000,subtype='FLOAT')
        r=schema.make_unit(a,'s0','你好',split='train');self.assertEqual(len(T._read_record(r)),1554)
    def test_gpu_and_bad_rank_refused_before_model_or_output(self):
        with patch.object(T,'load_official',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(RuntimeError,'not authorized'):
                T.run(config(),self.data,'unused',self.root/'gpu',device='cuda')
            with patch.dict(os.environ,{'WORLD_SIZE':'1','RANK':'1'}):
                with self.assertRaisesRegex(ValueError,'invalid rank'):
                    T.run(config(),self.data,'unused',self.root/'rank')
        self.assertFalse((self.root/'gpu').exists());self.assertFalse((self.root/'rank').exists())
    def test_exposure_counts_language_units_and_audio_separately(self):
        result=self.run_tiny(self.root/'one',stop=1)
        row=json.loads((self.root/'one/exposure_rank0.jsonl').read_text())
        self.assertEqual(result['exposure']['units'],sum(x['n_units'] for x in row['items']))
        self.assertAlmostEqual(result['exposure']['audio_seconds'],sum(x['duration_sec'] for x in row['items']))
        self.assertEqual(set(result['exposure']['units_by_type']),{'zh:han_characters_no_punctuation_v1'})
if __name__=='__main__':unittest.main()
