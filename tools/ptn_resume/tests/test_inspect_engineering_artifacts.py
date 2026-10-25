"""Bounded inspector helpers only, not real model/large-file acceptance."""
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from tools.ptn_resume import inspect_engineering_artifacts as I


class InspectorReadTests(unittest.TestCase):
    def test_hash_reads_actual_bytes_and_rejects_changed_pin(self):
        with tempfile.TemporaryDirectory()as temp:
            path=Path(temp).resolve()/'binary';path.write_bytes(b'first')
            pin=I.hash_file(path,0);self.assertEqual(pin['bytes'],5)
            path.write_bytes(b'second')
            with self.assertRaises(ValueError):I.load_bound(path,pin,None)

    def test_JSON_duplicate_nan_exponent_and_symlink_refused(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp).resolve()/'input.json'
            for raw in ('{"a":1,"a":2}','{"a":NaN}','{"a":1e999}'):
                p.write_text(raw)
                with self.assertRaises(ValueError):I.read_json(p)
            link=p.with_name('link');link.symlink_to(p)
            with self.assertRaises(ValueError):I.hash_file(link,0)

    def test_report_cannot_replace_input_or_existing_report(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp).resolve();p=root/'original.json';p.write_text('original')
            args=['--candidate2',str(root),'--candidate2-manifest-sha','a'*64,'--candidate3',str(root),
                '--candidate3-manifest-sha','b'*64,'--finite',str(root),'--injected',str(root),'--report',str(p),
                '--runner-sha',I.APPROVED_RUNNER_SHA]
            with self.assertRaises(ValueError):I.main(args)
            self.assertEqual(p.read_text(),'original')

    def test_runner_self_report_does_not_replace_external_approved_pin(self):
        I.validate_runner({'runner_sha256':I.APPROVED_RUNNER_SHA},I.APPROVED_RUNNER_SHA)
        for pin,identity in [('x'*64,{'runner_sha256':'x'*64}),('a'*64,{'runner_sha256':'a'*64}),
                (I.APPROVED_RUNNER_SHA,{'runner_sha256':'a'*64,'state_exact':True})]:
            with self.subTest(pin=pin),self.assertRaises(ValueError):I.validate_runner(identity,pin)


@unittest.skipUnless(importlib.util.find_spec('torch'),'existing CPU torch required')
class InspectorTensorTests(unittest.TestCase):
    def test_toy_geometry_momentum_scaler_and_RNG_checked(self):
        import torch
        import numpy as np
        torch.set_num_threads(1)
        model={'projection.weight':torch.ones(2,4),'bn.num_batches_tracked':torch.tensor(1)}
        opt={'param_groups':[{'params':[0]}],'state':{0:{'momentum_buffer':torch.ones(2,4)}}}
        scaler={'scale':131072.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000,'_growth_tracker':1841}
        out=I.validate_state(model,opt,scaler,[['projection.weight']],torch,counts=(2,1),groups=(1,),head=(2,4))
        self.assertEqual(out['ordered_parameter_count'],1)
        opt['state'][0]['momentum_buffer'][0,0]=float('nan')
        with self.assertRaises(ValueError):I.validate_state(model,opt,scaler,[['projection.weight']],torch,counts=(2,1),groups=(1,),head=(2,4))
        state=np.random.get_state();rng={'python':random.getstate(),'numpy':{'name':state[0],'keys':state[1].tolist(),
            'position':int(state[2]),'has_gauss':int(state[3]),'cached_gaussian':float(state[4])},'torch_cpu':torch.get_rng_state(),'torch_cuda':{}}
        I.validate_rng(rng,torch,None)
        with self.assertRaises(ValueError):I.validate_rng(rng,torch,0)

    def test_CPU_rng_structure_alone_cannot_approve_unrestorable_state(self):
        import torch
        import numpy as np
        n=np.random.get_state();rng={'python':random.getstate(),'numpy':{'name':n[0],'keys':n[1].tolist(),
            'position':int(n[2]),'has_gauss':int(n[3]),'cached_gaussian':float(n[4])},
            'torch_cpu':torch.zeros(16,dtype=torch.uint8),'torch_cuda':{}}
        before=torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError,'CPU RNG cannot be restored'):I.validate_rng(rng,torch,None)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
