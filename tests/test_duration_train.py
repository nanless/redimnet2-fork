import importlib.util as _deps
import unittest
if any(_deps.find_spec(p) is None for p in ['torch', 'soundfile']):
    raise unittest.SkipTest('optional CPU test dependencies unavailable: '+', '.join(['torch', 'soundfile']))
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from duration_ft import train as T
from duration_ft.data import inspect_record,seal_dataset,length_for_step,global_batch_indices
from test_duration_data import fixture

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(2,4);self.bn=nn.BatchNorm1d(4);self.drop=nn.Dropout(.2)
    def forward(self,x):return self.drop(self.bn(self.linear(torch.stack((x.mean(-1),x.square().mean(-1)),1))))
class TinyHead(nn.Module):
    def __init__(self,n):super().__init__();self.linear=nn.Linear(4,n);self.margin=.3
    def forward(self,x,y):
        z=self.linear(x);return z,nn.functional.cross_entropy(z,y)
def tiny_load(*args):return TinyBackbone(),{'embed_dim':4}
def tiny_head(dim,n,seed,args):
    with torch.random.fork_rng(devices=[]):torch.manual_seed(seed);return TinyHead(n)
def config():
    return {'seed':10,'head_seed':4,'batch_size':2,'total_steps':4,'checkpoint_every':2,
       'lengths':[1.,2.,3.,5.],'probabilities':[1/6,1/6,1/6,.5],
       'optimizer':{'lr':.001,'momentum':.9,'nesterov':True,'weight_decay':.0001},
       'head':{'margin_type':'C'},'official_sha256':T.OFFICIAL_SHA,
       'bn_policy':'frozen_statistics_trainable_affine','amp':False,'decoder':{}}

class TrainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);records=[]
        for i in range(3):
            obj,vad,_,_=fixture(self.root,f'A_s{i}/a.wav',f'A_s{i}')
            records.append(dict(inspect_record(obj,vad),split='train'))
        obj=seal_dataset(records,{f'A_s{i}':i for i in range(3)},{'split':'train'})
        self.data=self.root/'dataset.json';self.data.write_text(json.dumps(obj))
    def run_tiny(self,path,resume=None,stop=None,c=None):
        with patch.object(T,'load_official',tiny_load),patch.object(T,'build_head',tiny_head),patch.object(T,'decoder_identity',return_value={'test':'fixture'}):
            return T.run(c or config(),self.data,'unused',path,resume=resume,stop_after=stop)
    def test_resume_exact_model_optimizer_rng_order(self):
        self.run_tiny(self.root/'full');self.run_tiny(self.root/'split',stop=2)
        self.run_tiny(self.root/'split',resume=self.root/'split/train_step_2.pt')
        a=torch.load(self.root/'full/train_step_4.pt',weights_only=False);b=torch.load(self.root/'split/train_step_4.pt',weights_only=False)
        for name in ['state_dict','head']:
            for key in a[name]:torch.testing.assert_close(a[name][key],b[name][key],rtol=0,atol=0)
        for index,state in a['optimizer']['state'].items():
            for key,value in state.items():torch.testing.assert_close(value,b['optimizer']['state'][index][key],rtol=0,atol=0)
        self.assertEqual(a['exposure'],b['exposure']);self.assertEqual(a['initial_head_sha256'],b['initial_head_sha256'])
        aa=[json.loads(l) for l in (self.root/'full/exposure_rank0.jsonl').read_text().splitlines()]
        bb=[json.loads(l) for l in (self.root/'split/exposure_rank0.jsonl').read_text().splitlines()]
        for x,y in zip(aa,bb):
            for key in ['keys','crop_plans','seconds','loss','optimizer_update']:self.assertEqual(x[key],y[key])
        # Frozen BN running statistics stay at their pretrained values.
        self.assertTrue(torch.equal(a['state_dict']['bn.running_mean'],torch.zeros(4)))
        self.assertEqual(a['state_dict']['bn.num_batches_tracked'].item(),0)
        public=torch.load(self.root/'full/embedding_step_4.pt',weights_only=False)
        self.assertEqual(set(public),{'model_config','state_dict'});TinyBackbone().load_state_dict(public['state_dict'],strict=True)
    def test_resume_data_mismatch(self):
        self.run_tiny(self.root/'one',stop=2)
        c=config();c['seed']+=1
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            self.run_tiny(self.root/'one',resume=self.root/'one/train_step_2.pt',c=c)
    def test_world_mismatch(self):
        self.run_tiny(self.root/'one',stop=2)
        payload=torch.load(self.root/'one/train_step_2.pt',weights_only=False)
        model=T.FineTuneModel(TinyBackbone(),TinyHead(3));optimizer=torch.optim.SGD(model.parameters(),lr=.001)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            T.restore_checkpoint(payload,model,optimizer,scaler,config(),payload['dataset_id'],2,0)
    def test_m1_m2_same_head_and_order(self):
        c1=config();c1.update(lengths=[5.],probabilities=[1.]);a=self.run_tiny(self.root/'m1',stop=2,c=c1)
        b=self.run_tiny(self.root/'m2',stop=2)
        self.assertEqual(a['initial_head_sha256'],b['initial_head_sha256'])
        aa=[json.loads(l) for l in (self.root/'m1/exposure_rank0.jsonl').read_text().splitlines()]
        bb=[json.loads(l) for l in (self.root/'m2/exposure_rank0.jsonl').read_text().splitlines()]
        for x,y in zip(aa,bb):
            self.assertEqual(x['keys'],y['keys'])
            for long,short in zip(x['crop_plans'],y['crop_plans']):
                self.assertEqual(short['speech_axis_start'],long['speech_axis_start']+(80000-short['samples'])//2)
    def test_source_drift_rejected(self):
        self.run_tiny(self.root/'one',stop=2)
        with patch.object(T,'source_identity',return_value={'wrong':'sha'}):
            with self.assertRaisesRegex(ValueError,'source identity'):
                self.run_tiny(self.root/'one',resume=self.root/'one/train_step_2.pt')
    def test_output_protected(self):
        with self.assertRaisesRegex(ValueError,'protected output'):
            self.run_tiny('/root/experiments/redimnet2/b6-mix-official-20261019/new')
    def test_seed_never_touches_other_cuda_devices(self):
        with patch.object(torch.cuda,'manual_seed_all',side_effect=AssertionError('all GPUs touched')), \
             patch.object(torch.cuda,'is_initialized',return_value=False):
            T.seed_all(123)
            a=T.build_head(4,3,17,{'scale':32.,'margin':.3,'margin_type':'C','dynamic_margin':False})
            b=T.build_head(4,3,17,{'scale':32.,'margin':.3,'margin_type':'C','dynamic_margin':False})
            self.assertEqual(T.state_digest(a.state_dict()),T.state_digest(b.state_dict()))
        generators=[unittest.mock.Mock(),unittest.mock.Mock()]
        with patch.object(torch.cuda,'manual_seed_all',side_effect=AssertionError('all GPUs touched')), \
             patch.object(torch.cuda,'is_initialized',return_value=True), \
             patch.object(torch.cuda,'current_device',return_value=1), \
             patch.object(torch.cuda,'default_generators',generators):
            T.seed_all(123)
            generators[0].manual_seed.assert_not_called()
            generators[1].manual_seed.assert_called_once_with(123)

    def test_bad_rank_rejected_before_gpu_or_output(self):
        out=self.root/'invalid-rank'
        for env in [{'WORLD_SIZE':'1','RANK':'1','LOCAL_RANK':'0'},
                    {'WORLD_SIZE':'1','RANK':'-1','LOCAL_RANK':'0'},
                    {'WORLD_SIZE':'0','RANK':'0','LOCAL_RANK':'0'}]:
            with patch.dict('os.environ',env):
                with self.assertRaisesRegex(ValueError,'invalid distributed'):
                    T.run(config(),self.data,'unused',out,device='cuda')
            self.assertFalse(out.exists())

    def test_gpu_not_enabled(self):
        with patch.dict('os.environ',{'DURATION_FT_GPU_AUTHORIZED':'0'}):
            with self.assertRaisesRegex(RuntimeError,'GPU execution'):
                T.run(config(),self.data,'unused',self.root/'gpu',device='cuda')
if __name__=='__main__':unittest.main()
