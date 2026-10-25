import importlib.util as _deps
import unittest
if any(_deps.find_spec(p) is None for p in ['torch']):
    raise unittest.SkipTest('optional CPU test dependencies unavailable: '+', '.join(['torch']))
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from duration_ft.resume_guard import check_resume_output, ResumeRefused


class ResumeGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'output'
        self.root.mkdir()
        (self.root / '.training.lock').write_bytes(b'')
        self.step, self.world, self.batch = 2, 2, 1
        self.write_checkpoint(2)
        self.write_logs()

    def exposure(self, step, seconds=5.0):
        count = step * self.batch * self.world
        return {'samples': count, 'audio_seconds': count * seconds,
                'seconds_by_duration': {'5.0': count * seconds},
                'samples_by_duration': {'5.0': count}}

    def write_checkpoint(self, step, payload_step=None):
        payload_step = step if payload_step is None else payload_step
        value = {'schema': 'duration-training/v1', 'global_step': payload_step,
                 'world_size': self.world, 'config': {'batch_size': self.batch},
                 'sampler_global_position': payload_step * self.world * self.batch,
                 'exposure': self.exposure(payload_step), 'dummy_cpu_tensor': torch.zeros(2)}
        torch.save(value, self.root / f'train_step_{step}.pt')
        torch.save({'model_config': {}, 'state_dict': {}}, self.root / f'embedding_step_{step}.pt')

    def rows(self, rank):
        return [{'step': n, 'rank': rank, 'samples': self.batch, 'seconds': 5.0,
                 'optimizer_update': True, 'loss': 1.0, 'exposure_global': self.exposure(n)}
                for n in range(1, self.step + 1)]

    def write_logs(self):
        for rank in range(self.world):
            (self.root / f'exposure_rank{rank}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.rows(rank)))

    def check(self, **kwargs):
        return check_resume_output(self.root, self.root / 'train_step_2.pt', **kwargs)

    def tree_identity(self):
        return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                for p in self.root.iterdir() if p.is_file()}

    def test_normal_step2_same_output_passes_without_file_changes(self):
        before = self.tree_identity()
        result = self.check(world_size=2)
        self.assertEqual(result['global_step'], 2)
        self.assertEqual(result['published_steps'], [2])
        self.assertEqual(result['exposure_logs']['0']['rows'], 2)
        self.assertEqual(before, self.tree_identity())
        self.assertFalse(torch.cuda.is_initialized())

    def test_world1_supported(self):
        self.world = 1
        (self.root / 'exposure_rank1.jsonl').unlink()
        self.write_checkpoint(2)
        self.write_logs()
        self.assertEqual(self.check(world_size=1)['world_size'], 1)

    def test_older_step_refused(self):
        self.write_checkpoint(4)
        with self.assertRaisesRegex(ResumeRefused, 'older'):
            self.check()

    def test_uncheckpointed_log_update_refused(self):
        path = self.root / 'exposure_rank0.jsonl'
        row = dict(self.rows(0)[-1], step=3, exposure_global=self.exposure(3))
        with path.open('a') as stream:stream.write(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ResumeRefused, 'uncheckpointed'):
            self.check()

    def test_truncated_tail_refused(self):
        path = self.root / 'exposure_rank0.jsonl'
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ResumeRefused, 'truncated'):
            self.check()

    def test_duplicate_or_gap_exposure_refused(self):
        path = self.root / 'exposure_rank0.jsonl'
        original = path.read_bytes()
        for content in [original + original.splitlines(keepends=True)[-1], original.splitlines(keepends=True)[-1]]:
            path.write_bytes(content)
            with self.assertRaisesRegex(ResumeRefused, 'duplicate, missing'):
                self.check()

    def test_bad_json_blank_line_and_nonfinite_refused(self):
        path = self.root / 'exposure_rank0.jsonl'
        for raw in [b'{"step":1}\n{bad}\n', b'\n',
                    json.dumps(dict(self.rows(0)[0], loss=float('nan'))).encode() + b'\n',
                    b'{"step":1,"step":2}\n']:
            path.write_bytes(raw)
            with self.assertRaises(ResumeRefused):self.check()

    def test_missing_embedding_or_log_stream_refused(self):
        (self.root / 'embedding_step_2.pt').unlink()
        with self.assertRaisesRegex(ResumeRefused, 'unpaired'):self.check()
        self.write_checkpoint(2)
        (self.root / 'exposure_rank1.jsonl').unlink()
        with self.assertRaisesRegex(ResumeRefused, 'missing'):self.check()

    def test_new_output_and_foreign_checkpoint_refused_before_load(self):
        new = Path(self.tmp.name) / 'new'
        with mock.patch('torch.load') as loader:
            with self.assertRaises(ResumeRefused):check_resume_output(new, self.root / 'train_step_2.pt')
            new.mkdir()
            with self.assertRaisesRegex(ResumeRefused, 'same output'):check_resume_output(new, self.root / 'train_step_2.pt')
            loader.assert_not_called()

    def test_payload_filename_and_world_mismatch_refused(self):
        self.write_checkpoint(2, payload_step=1)
        with self.assertRaisesRegex(ResumeRefused, 'filename'):self.check()
        self.write_checkpoint(2)
        with self.assertRaisesRegex(ResumeRefused, 'world size'):self.check(world_size=1)

    def test_final_exposure_and_rank_history_mismatch_refused(self):
        path = self.root / 'exposure_rank1.jsonl'
        rows = self.rows(1);rows[-1]['exposure_global']['audio_seconds'] += 1
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with self.assertRaisesRegex(ResumeRefused, 'final exposure'):self.check()
        rows = self.rows(1);rows[0]['seconds'] = 1.0
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with self.assertRaisesRegex(ResumeRefused, 'histories'):self.check()

    def test_noncommitted_update_refused(self):
        path = self.root / 'exposure_rank0.jsonl'
        rows = self.rows(0);rows[-1]['optimizer_update'] = False
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with self.assertRaisesRegex(ResumeRefused, 'committed'):self.check()

    def test_checkpoint_tmp_and_previous_failure_preserved(self):
        path = self.root / 'train_step_4.pt.tmp.12';path.write_bytes(b'uncommitted')
        before = self.tree_identity()
        with self.assertRaisesRegex(ResumeRefused, 'temporary'):self.check()
        self.assertEqual(before, self.tree_identity())
        path.unlink();failure = self.root / 'failure_rank0_step2.txt';failure.write_text('past failure')
        with self.assertRaisesRegex(ResumeRefused, 'failure retained'):self.check()
        self.assertEqual(failure.read_text(), 'past failure')

    def test_prior_resume_attempt_identity_refused(self):
        (self.root / 'identity_start_step_2.json').write_text('{}')
        with self.assertRaisesRegex(ResumeRefused, 'attempt'):self.check()

    def test_symlink_resume_refused(self):
        original = self.root / 'train_step_2.pt';outside = Path(self.tmp.name) / 'outside.pt'
        original.rename(outside);original.symlink_to(outside)
        with self.assertRaisesRegex(ResumeRefused, 'same output'):self.check()

    def test_live_writer_refused_and_nonzero_rank_does_not_compete(self):
        import fcntl
        with (self.root / '.training.lock').open('rb') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ResumeRefused, 'writer'):self.check()
            self.assertFalse(self.check(rank=1)['writer_lock_checked'])

    def test_invalid_rank_cannot_bypass_writer_probe(self):
        for rank in [-1,True,2]:
            with self.subTest(rank=rank),self.assertRaises(ResumeRefused):self.check(rank=rank)
        with self.assertRaises(ResumeRefused):self.check(rank=1,world_size=1)


    def test_post_lock_recheck_closes_directory_advance_window(self):
        import fcntl
        self.assertEqual(self.check()['global_step'], 2)
        # Another completed attempt advances after precheck but before our writer lock.
        self.step = 4
        self.write_checkpoint(4)
        self.write_logs()
        with (self.root / '.training.lock').open('rb') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ResumeRefused, 'older'):
                self.check(writer_lock_held=True)

    def test_post_lock_normal_resume_does_not_conflict_with_own_lock(self):
        import fcntl
        with (self.root / '.training.lock').open('rb') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.check(writer_lock_held=True)
            self.assertEqual(result['global_step'], 2)
            self.assertTrue(result['writer_lock_held'])
            self.assertFalse(result['writer_lock_checked'])
            with self.assertRaises(ResumeRefused):self.check(rank=1,writer_lock_held=True)


if __name__ == '__main__':
    unittest.main()
