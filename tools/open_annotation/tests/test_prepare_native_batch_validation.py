"""原生合批准备的真实冻结、正式隔离与输出保护；不调用GPU。"""
import copy
import json
from pathlib import Path
import sys
import subprocess
import unittest
from unittest.mock import patch
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_native_batch_validation as N
from contracts import canonical_bytes, sha_file
import test_prepare_bounded_speed_validation as fixture
from test_maintenance_entries import external_contract


class NativePreparation(unittest.TestCase):
    def setUp(self):
        fixture.BoundedPreparation.setUp(self)
        base = Path(self.spec['base_contract']['path'])
        base.write_bytes(canonical_bytes(external_contract()))
        self.spec.update(schema=N.SCHEMA, target_uuids='auto2')
        self.spec['base_contract']['sha256'] = sha_file(base)

    def prepare(self, output=None):
        self.spec_path.write_bytes(canonical_bytes(self.spec))
        return N.prepare(self.spec_path, sha_file(self.spec_path), output or self.root/'native-new')

    def test_real_freezes_only_pilot_and_same_semantics_except_explicit_ASR_width(self):
        result = self.prepare()
        self.assertFalse(result['GPU_executed'])
        self.assertFalse(result['database_created'])
        self.assertEqual(result['worker_count'], 2)
        contracts = []
        for width, variant in zip((1, 4, 8), result['variants'].values()):
            self.assertEqual(set(variant['freeze']['contracts']), {'pilot_validation'})
            c = json.loads(Path(variant['freeze']['contracts']['pilot_validation']['path']).read_text())
            self.assertEqual(c['runtime']['asr_batching']['max_requests'], width)
            self.assertEqual(c['runtime']['asr_compute'], {'dtype':'float32','attention':'sdpa'})
            self.assertEqual(c['runtime']['max_batch'], 1)
            self.assertIn('native_transformers_batch.py', c['engine_source_files'])
            contracts.append(c)
        for c in contracts[1:]:
            for key in ('models','source_files','alignment','quality_policy','engine_source_files'):
                self.assertEqual(c[key], contracts[0][key])
            left, right = copy.deepcopy(c['runtime']), copy.deepcopy(contracts[0]['runtime'])
            left['asr_batching'].pop('max_requests');right['asr_batching'].pop('max_requests')
            self.assertEqual(left, right)
        self.assertFalse(list((self.root/'native-new').rglob('*.sqlite')))

    def test_old_profile_auto8_duplicate_and_unknown_batch_refused(self):
        for key, value in [('target_uuids','auto8'), ('profiles',['asr-fp32-eager','asr-fp32-sdpa']),
                           ('profiles',[N.DEFAULT_PROFILES[0]]*3),
                           ('profiles',[N.DEFAULT_PROFILES[0],'native-fp32-sdpa-b16'])]:
            before = copy.deepcopy(self.spec)
            self.spec[key] = value
            with self.subTest(value=value), self.assertRaises(RuntimeError):self.prepare()
            self.assertFalse((self.root/'native-new').exists())
            self.spec = before

    def test_full_library_denominator_refused_before_freeze(self):
        self.spec['source_count'] = 11521178
        with patch.object(N.freezer,'freeze') as freeze:
            with self.assertRaisesRegex(RuntimeError,'finite'):self.prepare()
            freeze.assert_not_called()

    def test_per_source_features_are_explicit_and_frozen_for_every_width(self):
        import native_b1_feature_padding as features
        self.spec['asr_preprocessing'] = {'policy': 'official_b1_features_then_pad/v1'}
        # 仅验证准备器怎样固定七类来源；这些空实现不具备任何模型资格。
        base_path = Path(self.spec['base_contract']['path'])
        base = json.loads(base_path.read_text())
        model = self.root/'SYNTHETIC_NOT_MODEL'; model.mkdir()
        config = model/'preprocessor_config.json'; config.write_text('{}\n')
        base['models']['asr']['path'] = str(model)
        base['models']['asr']['files']['preprocessor_config.json'] = {'bytes':config.stat().st_size,'sha256':sha_file(config)}
        pins = {'schema':'annotation-official-b1-feature-sources/v1','files':{},'processor_config':features.file_ref(config)}
        for role in features.SOURCE_ROLES:
            source = self.root/(role+'.py'); source.write_text('# synthetic source binding; not a model implementation\n')
            pins['files'][role] = features.file_ref(source)
            if role in ('qwen_processor','sdk_normalizer'):base['source_files'][str(source)] = sha_file(source)
        base_path.write_bytes(canonical_bytes(base))
        self.spec['base_contract']['sha256'] = sha_file(base_path)
        self.spec['asr_preprocessing_sources'] = pins
        prepared = self.prepare()
        self.assertEqual(prepared['asr_preprocessing'], self.spec['asr_preprocessing'])
        self.assertEqual(prepared['asr_preprocessing_sources'], pins)
        for variant in prepared['variants'].values():
            c = json.loads(Path(variant['freeze']['contracts']['pilot_validation']['path']).read_text())
            self.assertEqual(c['runtime']['asr_preprocessing'], self.spec['asr_preprocessing'])
            self.assertEqual(c['asr_preprocessing_sources'], pins)
            self.assertIn('native_b1_feature_padding.py', c['engine_source_files'])
            self.assertEqual(c['purpose'], 'pilot_validation')
        self.assertFalse(prepared['GPU_executed'])

    def test_unknown_or_extra_feature_policy_refused_before_copying_sources(self):
        for policy in ({'policy':'batch_waveform_padding/v0'},
                       {'policy':'official_b1_features_then_pad/v1','ignore_mismatch':True}, {}):
            self.spec['asr_preprocessing'] = policy
            with self.subTest(policy=policy), patch.object(N.freezer, 'freeze') as freeze:
                with self.assertRaisesRegex(RuntimeError, 'feature policy'): self.prepare()
                freeze.assert_not_called()

    def test_new_feature_policy_without_source_pins_is_not_prepared(self):
        self.spec['asr_preprocessing'] = {'policy':'official_b1_features_then_pad/v1'}
        with patch.object(N.freezer, 'freeze') as freeze:
            with self.assertRaisesRegex(RuntimeError, 'source pins'):self.prepare()
            freeze.assert_not_called()

    def test_production_and_PTN_outputs_refused(self):
        for root in self.spec['protected_roots']:
            with self.subTest(root=root), self.assertRaisesRegex(RuntimeError,'historical'):
                self.prepare(Path(root)/'native')

    def test_parent_traversal_cannot_bypass_protected_output_or_lock_roots(self):
        protected = Path(self.spec['protected_roots'][0])
        cases = [('output', self.root/'fresh'/'..'/protected.name/'new'),
                 ('protected_roots', [str(protected/'x'/'..')]),
                 ('device_lock_root', str(self.root/'locks'/'x'/'..'))]
        for field, value in cases:
            before = copy.deepcopy(self.spec)
            if field != 'output': self.spec[field] = value
            with self.subTest(field=field), patch.object(N.freezer, 'freeze') as freeze:
                with self.assertRaisesRegex(RuntimeError, 'physical absolute path'):
                    self.prepare(value if field == 'output' else None)
                freeze.assert_not_called()
            self.spec = before

    def test_deployed_layout_real_subprocess_freezes_sibling_without_GPU(self):
        task=self.root/'experiments'/'native-pilot';task.mkdir(parents=True)
        code=task/'build-code'
        N.freezer.freeze(self.spec['base_contract']['path'],self.spec['base_contract']['sha256'],code,'native-deployed-test')
        self.spec_path.write_bytes(canonical_bytes(self.spec))
        argv=[sys.executable,'-I','-B']+(['-O']if sys.flags.optimize else[])
        result=subprocess.run([*argv,str(code/'prepare_native_batch_validation.py'),
            '--spec',str(self.spec_path),'--spec-sha',sha_file(self.spec_path),'--output',str(task/'screen')],
            capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        prepared=json.loads((task/'screen'/'prepared.json').read_text())
        self.assertEqual(prepared['worker_count'],2)
        self.assertEqual(prepared['execution_order'],N.DEFAULT_PROFILES)
        for variant in prepared['variants'].values():
            frozen=Path(variant['freeze']['frozen_copy'])
            self.assertEqual(sha_file(frozen/'prepare_native_batch_validation.py'),sha_file(code/'prepare_native_batch_validation.py'))
            self.assertEqual(set(variant['freeze']['contracts']),{'pilot_validation'})
        self.assertFalse(prepared['GPU_executed'])


if __name__ == '__main__':unittest.main()
