"""Synthetic source clocks only; no audio, model, GPU or external samples."""
from copy import deepcopy
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_windows import units_from_windows
from source_clock import LEGACY_MAPPING, TIME_MAPPING, sample_bounds
import schema
from test_quality_schema import fixture


def window(items, sr=16000, mapping=TIME_MAPPING, repaired=False):
    return {'window_id': 'w0', 'context_start_sample': 0, 'context_end_sample': 20*sr,
            'owned_start_sample': 0, 'owned_end_sample': 20*sr, 'time_mapping': mapping,
            'alignment': {'repair_changed': repaired, 'items': [
                {'text': chr(65+i), 'start_time': a, 'end_time': b} for i,(a,b) in enumerate(items)]}}


class SourceClockTests(unittest.TestCase):
    def test_decimal_touch_has_no_invented_sample_overlap(self):
        for boundary in (16.08, 16.24, 1.001):
            with self.subTest(boundary=boundary):
                w=window([(boundary-.08,boundary),(boundary,boundary+.08)])
                units,flags=units_from_windows([w],{'sample_rate':16000,'frames':320000})
                self.assertEqual(flags,[])
                self.assertEqual(units[0]['end_sample'],units[1]['start_sample'])
                old=deepcopy(w);old['time_mapping']=LEGACY_MAPPING
                _,old_flags=units_from_windows([old],{'sample_rate':16000,'frames':320000})
                self.assertTrue(any(x['reason']=='cross_window_or_native_overlap_unresolved' for x in old_flags))

    def test_real_overlap_is_not_tolerated_or_clipped(self):
        w=window([(16.0,16.241),(16.24,16.32)])
        units,flags=units_from_windows([w],{'sample_rate':16000,'frames':320000})
        self.assertEqual(units[0]['end_sample']-units[1]['start_sample'],16)
        self.assertTrue(any(x['reason']=='cross_window_or_native_overlap_unresolved' for x in flags))

    def test_fractional_grid_coverage_remains_overlap(self):
        # 1 ms is 44.1 source samples. Conservative floor/ceil really overlap;
        # unlike binary arithmetic drift this must remain unresolved.
        w=window([(0,.001),(.001,.002)],sr=44100)
        units,flags=units_from_windows([w],{'sample_rate':44100,'frames':882000})
        self.assertEqual((units[0]['end_sample'],units[1]['start_sample']),(45,44))
        self.assertTrue(flags)

    def test_repaired_observations_keep_unknown_and_native_bytes(self):
        args=list(fixture());w=args[4][0]
        w['alignment']['repair_changed']=True
        w['alignment']['native_timestamps_ms']=[100,300,800,1200]
        for u in args[5]:u['repair_applied']=True
        before=deepcopy(w['alignment']);r=schema.build_record(*args)
        self.assertEqual(r['decision']['state'],'unknown')
        self.assertIn('window_0:alignment_repaired_or_repair_unknown',r['decision']['reasons'])
        self.assertEqual(before,r['windows'][0]['alignment'])
        self.assertFalse(r['decision']['usable_for_training'])

    def test_legacy_record_is_validated_by_its_declared_mapping(self):
        args=list(fixture());args[4][0]['time_mapping']=LEGACY_MAPPING
        r=schema.build_record(*args);self.assertEqual(schema.validate_record(r),[])
        self.assertEqual(r['quality_axes']['source_clock']['mapping'],LEGACY_MAPPING)
        self.assertEqual(args[4][0]['time_mapping'],LEGACY_MAPPING)

    def test_quality_axes_package_and_script_imports_work_from_independent_cwd(self):
        repo=Path(__file__).resolve().parents[3]
        for package in (True,False):
            with self.subTest(package=package),tempfile.TemporaryDirectory()as cwd:
                root=repo if package else repo/'tools/open_annotation'
                statement='import tools.open_annotation.quality_axes as q'if package else'import quality_axes as q'
                code=("import sys;sys.path.insert(0,sys.argv[1]);"+statement+";"
                      "assert q.heldout_source({'roles':['evaluation:VOICES']});"
                      "assert not q.heldout_source({'original_splits':['provider_train']});"
                      "assert 'torch'not in sys.modules;"
                      +( "assert 'source_clock'not in sys.modules;"if package else"assert 'source_clock'in sys.modules;")
                      +"print('dual-entry-OK')")
                env={k:v for k,v in os.environ.items()if k!='PYTHONPATH'};env['PYTHONDONTWRITEBYTECODE']='1'
                result=subprocess.run([sys.executable,'-B','-c',code,str(root)],cwd=cwd,env=env,capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr);self.assertIn('dual-entry-OK',result.stdout)

    def test_exact_integer_and_invalid_inputs(self):
        self.assertEqual(sample_bounds(16.24,16.32,16000),(259840,261120))
        self.assertEqual(sample_bounds(.07,.14,44100),(3087,6174))
        for args in ((True,1,16000),(.1,float('nan'),16000),(.1,.2,True),(.1,.2,0)):
            with self.subTest(args=args),self.assertRaises(ValueError):sample_bounds(*args)
        with self.assertRaises(ValueError):sample_bounds(.1,.2,16000,'unknown')

    def test_window_cannot_switch_mapping_declared_by_producer(self):
        args=list(fixture());args[4][0]['time_mapping']=LEGACY_MAPPING
        args[2]['provenance']['runtime']={'boundary_mapping':TIME_MAPPING}
        r=schema.build_record(*args)
        self.assertIn('window:source_clock_not_bound_to_producer_runtime',schema.validate_record(r))
        self.assertIn('window_0:source_clock_not_bound_to_producer_runtime',r['decision']['reasons'])
        args[2]['provenance']['runtime']['boundary_mapping']=LEGACY_MAPPING
        r=schema.build_record(*args);self.assertEqual(schema.validate_record(r),[])


if __name__=='__main__':unittest.main()
