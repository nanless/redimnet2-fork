"""Synthetic full failure capture; no official models, media or GPU required."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'applications/speaker_rec'))
from wespeaker_lite.utils import numerical_failure as N
from wespeaker_lite.utils.executor import _check_training_numerics


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,4)
        self.bn = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(.4)
        self.projection = nn.Linear(4,2)
        self.register_buffer('scratch', torch.zeros(4), persistent=False)
    def forward(self, wave, lengths=None):
        value = self.dropout(self.bn(self.linear(wave)))
        self.scratch.copy_(value.detach().mean(0))
        return value


def seed(value=17):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def prepared():
    return torch.arange(32,dtype=torch.float32).reshape(8,4)/20, torch.arange(8)%2


def state():
    seed(); model=Toy().train(); optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    wave, targets=prepared()
    loss=nn.functional.cross_entropy(model.projection(model(wave)),targets)
    loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    return model,optimizer,torch.amp.GradScaler('cpu',enabled=False)


def payload(model,optimizer,scaler,capture,wave,targets,embed):
    return N.make_payload(model=model,optimizer=optimizer,scaler=scaler,capture=capture,
        wav=wave,targets=targets,embeds=embed,loss=embed.sum()*float('nan'),
        utts=['synthetic-%d'%x for x in range(len(targets))],
        original_batch={'wav':wave,'label':targets,'key':['synthetic']*len(targets)},
        location={'epoch':2,'batch':3,'rank':0,'world_size':1,'absolute_iteration':7},enable_amp=False)


def distributed_fault_worker(rank, port, directory, fail_write):
    """Real loopback process group: a finite peer must also export, then stop."""
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:%d' % port,
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=30))
    try:
        model,opt,scaler=state()
        ddp=torch.nn.parallel.DistributedDataParallel(model)
        with torch.no_grad(): model.bn.running_mean.fill_(3 if rank==0 else 17)
        wave,targets=prepared()
        with N.ForwardCapture(ddp,targets=targets) as capture:embed=ddp(wave)
        # Bare hook must see rank0's broadcast, including on rank1.
        assert torch.equal(capture.state['buffers']['bn.running_mean'],torch.full((4,),3.))
        loss=embed.sum() if rank else embed.sum()*float('nan')
        original=N.write_failure_bundle
        def write(*args,**kwargs):
            if fail_write and rank==1:raise OSError('synthetic rank1 write failure')
            return original(*args,**kwargs)
        stopped=False
        with patch.object(N,'write_failure_bundle',side_effect=write):
            try:
                _check_training_numerics(wave,embed,loss,epoch=2,batch_index=3,rank=rank,
                    optimizer=opt,scaler=scaler,model=ddp,targets=targets,capture_context=capture,
                    diagnostic_dir=directory,enable_amp=False)
            except FloatingPointError:stopped=True
        assert stopped
        Path(directory,'rank%d-returned.json'%rank).write_text(json.dumps({'stopped':True}))
    finally:
        dist.destroy_process_group()


class FailureCaptureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def captured(self):
        model,optimizer,scaler=state();wave,targets=prepared()
        with N.ForwardCapture(model,targets=targets) as capture:embed=model(wave)
        return model,optimizer,scaler,wave,targets,capture,embed

    def test_full_capture_replays_dropout_BN_and_optimizer(self):
        model,opt,scaler,wave,targets,capture,expected=self.captured()
        dump=payload(model,opt,scaler,capture,wave,targets,expected)
        self.assertTrue(dump['replay_complete'])
        self.assertFalse(torch.equal(dump['buffers_pre_forward']['bn.running_mean'],
                                     dump['buffers_post_forward']['bn.running_mean']))
        clone=Toy().train();new_opt=torch.optim.SGD(clone.parameters(),lr=.4,momentum=.9)
        inputs=N.restore_pre_forward(clone,dump,optimizer=new_opt,
                                     scaler=torch.amp.GradScaler('cpu',enabled=False))
        self.assertTrue(torch.equal(clone(inputs['wav']),expected))
        self.assertTrue(new_opt.state)
        self.assertEqual(new_opt.param_groups[0]['lr'],.01)
        self.assertTrue(torch.equal(inputs['targets'],targets))
        self.assertEqual(set(dump['model_pre_forward']),set(model.state_dict()))
        self.assertIn('scratch',dump['buffers_pre_forward'])

    def test_snapshot_does_not_consume_python_numpy_or_torch_rng(self):
        model,_,_=state();wave,targets=prepared();before=N.rng_state(model)
        with N.ForwardCapture(model,targets=targets) as capture:
            capture._capture(model,(wave,),{})  # Capture only, do not invoke stochastic forward.
        observed=(random.random(),float(np.random.rand()),torch.rand(4))
        N.restore_rng(before);expected=(random.random(),float(np.random.rand()),torch.rand(4))
        self.assertEqual(observed[:2],expected[:2]);self.assertTrue(torch.equal(observed[2],expected[2]))

    def test_prepared_targets_soft_labels_lengths_are_immutable_and_multiple_calls_refused(self):
        model,opt,scaler=state();wave,targets=prepared()
        mask=nn.functional.one_hot(targets,2).float();lengths=torch.full((8,),4)
        with N.ForwardCapture(model,targets=targets,target_mask=mask,lengths=lengths) as capture:
            embed=model(wave,lengths=lengths)
        targets.fill_(9);mask.fill_(0);lengths.fill_(1)
        dump=payload(model,opt,scaler,capture,wave,targets,embed)
        self.assertTrue(dump['replay_complete'])
        self.assertTrue(torch.equal(dump['inputs']['targets'],torch.arange(8)%2))
        self.assertTrue(torch.equal(dump['inputs']['lengths'],torch.full((8,),4)))
        self.assertTrue(torch.equal(dump['inputs']['target_mask'].sum(1),torch.ones(8)))
        with N.ForwardCapture(model,targets=targets) as capture:
            model(wave);embed=model(wave)
        self.assertFalse(payload(model,opt,scaler,capture,wave,targets,embed)['replay_complete'])

    def test_buffer_resize_is_incomplete_and_bad_rng_mapping_does_not_change_cpu_rng(self):
        model,opt,scaler,wave,targets,capture,embed=self.captured()
        model.scratch.resize_(5)
        dump=payload(model,opt,scaler,capture,wave,targets,embed)
        self.assertFalse(dump['replay_complete'])
        self.assertIn('registered buffer shape/dtype changed during forward',dump['replay_incomplete_reasons'])
        rng=N.rng_state(model);torch.rand(5);before=torch.get_rng_state().clone()
        with self.assertRaises(ValueError):N.restore_rng(rng,{False:0})
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_finite_path_never_serializes_model_optimizer_or_files(self):
        model,opt,scaler=state();wave,targets=prepared()
        with patch.object(model,'state_dict',side_effect=AssertionError('no full model copy')),\
             patch.object(opt,'state_dict',side_effect=AssertionError('no optimizer copy')),\
             patch.object(N,'write_failure_bundle',side_effect=AssertionError('no disk write')):
            with N.ForwardCapture(model,targets=targets) as capture:embed=model(wave)
            _check_training_numerics(wave,embed,embed.sum(),epoch=1,batch_index=1,rank=0,
                optimizer=opt,scaler=scaler,model=model,targets=targets,capture_context=capture,
                diagnostic_dir=self.root)
        self.assertEqual(list(self.root.iterdir()),[])
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
        self.assertNotIn('model',capture.state)
        self.assertGreater(capture.state['snapshot_tensor_bytes'],0)
        capture.release();self.assertIsNone(capture.state)

    def test_immutable_payload_contains_complete_fields_and_safe_load(self):
        model,opt,scaler,wave,targets,capture,embed=self.captured()
        dump=payload(model,opt,scaler,capture,wave,targets,embed)
        result=N.write_failure_bundle(self.root,payload=dump,summary={'rank':0},stem='event')
        event=Path(result['directory']);raw=(event/'payload.pt').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),result['payload_sha256'])
        loaded=torch.load(event/'payload.pt',weights_only=True,map_location='cpu')
        for name in ['model_pre_forward','model_post_forward','optimizer_state','scaler_state',
                     'targets','wav','rng_pre_forward','rng_at_failure_capture',
                     'buffers_pre_forward','buffers_post_forward','runtime_identity']:
            self.assertIn(name,loaded)
        self.assertNotIn('model',loaded);self.assertNotIn('epoch',loaded)
        self.assertFalse(loaded['training_resume_checkpoint'])
        before={p.name:p.read_bytes() for p in event.iterdir()}
        with self.assertRaises(FileExistsError):
            N.write_failure_bundle(self.root,payload=dump,summary={},stem='event')
        self.assertEqual(before,{p.name:p.read_bytes() for p in event.iterdir()})

    def test_partial_save_has_no_complete_marker_and_keeps_original_error(self):
        model,opt,scaler,wave,targets,capture,embed=self.captured()
        dump=payload(model,opt,scaler,capture,wave,targets,embed)
        def broken(value,stream):
            stream.write(b'partial evidence');raise OSError('synthetic disk fault')
        with patch.object(N.torch,'save',side_effect=broken):
            with self.assertRaises(OSError):N.write_failure_bundle(self.root,payload=dump,summary={},stem='partial')
        self.assertFalse((self.root/'partial/complete.json').exists())
        self.assertEqual((self.root/'partial/payload.pt.part').read_bytes(),b'partial evidence')
        self.assertIn('synthetic disk fault',(self.root/'partial/write_error.json').read_text())

    def test_parameter_mutation_or_capture_failure_is_not_replay_complete(self):
        model,opt,scaler,wave,targets,capture,embed=self.captured()
        with torch.no_grad():model.linear.weight.add_(1)
        dump=payload(model,opt,scaler,capture,wave,targets,embed)
        self.assertFalse(dump['replay_complete'])
        with self.assertRaises(ValueError):N.restore_pre_forward(Toy(),dump)
        with N.ForwardCapture(model,targets=targets,max_snapshot_bytes=1) as failed:embed=model(wave)
        self.assertFalse(failed.complete);self.assertIn('budget',failed.error)
        self.assertFalse(payload(model,opt,scaler,failed,wave,targets,embed)['replay_complete'])

    def test_actual_bad_guard_exports_and_never_updates(self):
        model,opt,scaler,wave,targets,capture,embed=self.captured()
        before=copy.deepcopy(model.state_dict());before_opt=copy.deepcopy(opt.state_dict())
        with self.assertRaises(FloatingPointError):
            _check_training_numerics(wave,embed,embed.sum()*float('nan'),epoch=2,batch_index=3,rank=0,
                optimizer=opt,scaler=scaler,model=model,targets=targets,capture_context=capture,
                diagnostic_dir=self.root,original_batch={'wav':wave,'label':targets},
                absolute_iteration=7,epoch_iter=5,enable_amp=False)
        folders=list(self.root.iterdir());self.assertEqual(len(folders),1)
        self.assertTrue((folders[0]/'complete.json').is_file())
        coordination=json.loads((folders[0]/'coordination.json').read_text())
        self.assertTrue(coordination['all_rank_replay_complete'])
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
        self.assertEqual(before_opt['param_groups'],opt.state_dict()['param_groups'])

    def test_incomplete_snapshot_is_global_gate_failure_even_if_numbers_finite(self):
        model,opt,scaler=state();wave,targets=prepared()
        with N.ForwardCapture(model,targets=targets,max_snapshot_bytes=1) as capture:embed=model(wave)
        with self.assertRaises(FloatingPointError):
            _check_training_numerics(wave,embed,embed.sum(),epoch=1,batch_index=1,rank=0,
                optimizer=opt,scaler=scaler,model=model,targets=targets,capture_context=capture,
                diagnostic_dir=self.root)
        complete=next(self.root.glob('*/complete.json'))
        self.assertFalse(json.loads(complete.read_text())['replay_complete'])

    def run_distributed_fault(self,fail_write):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        mp.spawn(distributed_fault_worker,args=(port,str(self.root),fail_write),nprocs=2,join=True)
        for rank in range(2):self.assertTrue((self.root/('rank%d-returned.json'%rank)).is_file())
        files=list(self.root.glob('*/coordination.json'))
        self.assertEqual(len(files),1 if fail_write else 2)
        for file in files:
            record=json.loads(file.read_text());self.assertTrue(record['coherent'])
            self.assertEqual(record['all_local_payloads_published'],not fail_write)
            self.assertEqual(record['all_rank_replay_complete'],not fail_write)
            self.assertFalse(record['participants'][0]['numerical_finite'])
            self.assertTrue(record['participants'][1]['numerical_finite'])
            if fail_write:self.assertIn('rank1 write failure',record['participants'][1]['write_error'])

    def test_real_two_rank_finite_peer_exports_after_DDP_broadcast(self):
        self.run_distributed_fault(False)

    def test_real_two_rank_one_writer_failure_still_coordinates_stop(self):
        self.run_distributed_fault(True)


if __name__=='__main__':unittest.main()
