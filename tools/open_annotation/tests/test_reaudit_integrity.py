"""第二轮只用合成文件复查缓存路径、完整计账及原生证据。"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts import Refused, canonical_bytes, sha_file
from sidecar import StageCache, SidecarStore
from contracts import safe_join, relative_path
import gpu_launch
from quality_axes import native_observation_errors
from report import verify_run
from work_queue import WorkQueue, SLOT_BYTES, decode_completion_slot, encode_completion_slot
import test_audit_outputs as audit_fixture
from test_ultra_observation import observation_fixture


class CachePathReauditTests(unittest.TestCase):
    def test_reserved_or_NUL_relative_names_are_not_directories(self):
        for raw in ('.','..','bad\x00name'):
            with self.subTest(raw=repr(raw)):
                with self.assertRaises(Refused):relative_path(raw)

    def test_version_root_and_internal_output_alias_refused_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'labels';output.mkdir();outside=root/'outside';outside.mkdir()
            (output/'version').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(Refused):SidecarStore(output,{'version':'version'},'a'*64)
            self.assertEqual(list(outside.iterdir()),[])
            owned=root/'owned';owned.mkdir();real=owned/'real';real.mkdir()
            (owned/'alias').symlink_to(real,target_is_directory=True)
            with self.assertRaises(Refused):safe_join(owned,'alias/new.json')

    def test_cache_root_symlink_cannot_publish_outside_owned_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);outside=root/'outside';outside.mkdir();link=root/'cache'
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(Refused):
                StageCache(link).put('asr', {'contract':'synthetic'}, {'text':'synthetic'})
            self.assertEqual(list(outside.iterdir()), [])

    def test_cache_ancestor_symlink_cannot_read_or_write_external_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);outside=root/'outside';owned=root/'owned';owned.mkdir()
            inputs={'contract':'synthetic','PCM':'same-fixture'}
            StageCache(outside).put('asr',inputs,{'text':'synthetic'})
            (owned/'asr').symlink_to(outside/'asr',target_is_directory=True)
            with self.assertRaises(Refused):StageCache(owned).get('asr',inputs)
            with self.assertRaises(Refused):StageCache(owned).put('asr',{'other':'fixture'}, {'text':'synthetic'})


class NativeObservationReauditTests(unittest.TestCase):
    def test_native_logits_digest_shape_and_index_bounds_are_required(self):
        for change in ('missing_hash','malformed_hash','wrong_rows','short_vocabulary','slot_outside_input'):
            with self.subTest(change=change):
                a=observation_fixture();forward=a['native_observation']['thinker_forwards'][0]
                if change=='missing_hash':forward.pop('slot_logits_sha256')
                elif change=='malformed_hash':forward['slot_logits_sha256']='not-a-sha'
                elif change=='wrong_rows':forward['slot_logit_shape']=[1,20]
                elif change=='short_vocabulary':forward['slot_logit_shape']=[4,2]
                else:forward['input_ids_shape']=[1,3]
                self.assertTrue(native_observation_errors(a), 'contradictory native evidence must remain uncertain')


class ParentPreflightReauditTests(unittest.TestCase):
    def test_invalid_contract_is_refused_before_model_files_or_GPU_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'task';base.mkdir();cp=root/'contract.json';ip=root/'index.json'
            cp.write_text('{"schema":"deliberately-invalid-contract"}');ip.write_text('{}')
            args=SimpleNamespace(max_runtime_seconds=60,bundle_sha='a'*64,contract=str(cp),contract_sha=sha_file(cp),
                shard_index=str(ip),index_sha=sha_file(ip),run_root=str(base),device_lock_root=str(root/'locks'),
                output=str(base/'resource'),queue_root=str(base/'queue'),output_root=str(base/'labels'))
            with patch.dict(os.environ,{'WORLD_SIZE':'1','RANK':'0'},clear=True), \
                 patch.object(gpu_launch,'verify_bundle',return_value={}), \
                 patch.object(gpu_launch,'ensure_alias'), \
                 patch.object(gpu_launch,'model_facts',side_effect=AssertionError('model/resource admission reached before contract validation')) as models:
                with self.assertRaisesRegex(RuntimeError,'production contract schema'):
                    gpu_launch.run(args)
                models.assert_not_called()


class FullReportReauditTests(unittest.TestCase):
    def setUp(self):
        self.fixture=audit_fixture.AuditTests(methodName='test_clean_tree_passes_and_cache_is_explicitly_excluded')
        self.fixture.setUp()

    def tearDown(self):self.fixture.tearDown()

    def result(self):
        f=self.fixture
        return verify_run(f.cp,f.ip,f.q,f.output,full=True)

    def slot(self):
        path=self.fixture.q/'completion_map.bin'
        return path,decode_completion_slot(path.read_bytes()[:SLOT_BYTES])

    def bind_chunk_again(self, value):
        f=self.fixture;path=next((f.q/'completed').glob('*.json'))
        path.write_bytes(canonical_bytes(value))
        slot_path,slot=self.slot();slot['commit_sha256']=sha_file(path)
        slot_path.write_bytes(encode_completion_slot(slot))

    def test_full_report_recomputes_quality_counts_from_immutable_sidecars(self):
        self.assertTrue(self.result()['passed'])
        path,slot=self.slot();slot['decision_counts']={'candidate':2,'unknown':0,'failed':0}
        path.write_bytes(encode_completion_slot(slot))
        self.assertFalse(self.result()['passed'], 'full audit must not trust a contradictory aggregate')

    def test_full_report_checks_complete_source_commit_descriptor(self):
        f=self.fixture;commit=Path(str(f.first)+'.commit.json');original=json.loads(commit.read_bytes())
        chunk=next((f.q/'completed').glob('*.json'));receipt=json.loads(chunk.read_bytes())
        for field,value in [('schema','wrong-schema'),('source_id','src_'+'f'*64),('bytes',-1)]:
            with self.subTest(field=field):
                descriptor=deepcopy(original);descriptor[field]=value;commit.write_bytes(canonical_bytes(descriptor))
                current=deepcopy(receipt)
                source=next(x for x in current['sources'] if x['sidecar_relpath'].endswith('audio0.wav.annotation.json'))
                source['commit_sha256']=sha_file(commit);self.bind_chunk_again(current)
                self.assertFalse(self.result()['passed'], 'full descriptor identity is part of publication')

    def test_full_report_checks_chunk_schema_id_and_source_count(self):
        path=next((self.fixture.q/'completed').glob('*.json'));original=json.loads(path.read_bytes())
        for field,value in [('schema','wrong-schema'),('shard_id','foreign-chunk'),('source_count',999)]:
            with self.subTest(field=field):
                current=deepcopy(original);current[field]=value;self.bind_chunk_again(current)
                self.assertFalse(self.result()['passed'], 'chunk identity must match fixed index')

    def test_queue_completion_file_symlink_refused_on_reopen(self):
        f=self.fixture;slot=f.q/'completion_map.bin';outside=f.root/'outside-completion.bin'
        outside.write_bytes(slot.read_bytes());before=outside.read_bytes();slot.unlink();slot.symlink_to(outside)
        with self.assertRaises(Refused):WorkQueue(f.ip,sha_file(f.ip),f.q,sha_file(f.cp))
        self.assertEqual(outside.read_bytes(),before)

    def test_queue_claim_directory_symlink_cannot_create_external_lock(self):
        f=self.fixture;queue=WorkQueue(f.ip,sha_file(f.ip),f.root/'new-queue',sha_file(f.cp))
        outside=f.root/'outside-locks';outside.mkdir();(queue.root/'claims').symlink_to(outside,target_is_directory=True)
        try:
            with self.assertRaises(Refused):
                claim=queue.claim_next('synthetic')
                if claim is not None:claim.close()
            self.assertEqual(list(outside.iterdir()),[])
        finally:
            # The original deliberately vulnerable implementation may have made
            # a fixture lock, but no file outside this temporary tree is used.
            pass


if __name__=='__main__':unittest.main()
