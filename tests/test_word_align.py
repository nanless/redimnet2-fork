"""Synthetic alignment audits only; never load ASR, aligner weights or CUDA."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np

from word_ft.align import (AlignmentRefused, NativeQwenCPUBackend, align_training_record,
                           build_candidate_spans, capture_native_alignment, hanzi_text,
                           identity_digest, validate_character_alignment, matched_prefix_spans)

try:
    import soundfile
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


class FakeProcessor:
    def fix_timestamp(self, data):
        return np.maximum.accumulate(data).astype(int).tolist()


class FakeNative:
    timestamp_segment_time = 80
    def __init__(self, raw=None, fail=False, observe=True):
        self.aligner_processor = FakeProcessor()
        self.raw = np.array(raw if raw is not None else [80, 240, 400, 560, 800, 960])
        self.fail, self.observe = fail, observe

    def align(self, audio, text, language):
        fixed = self.aligner_processor.fix_timestamp(self.raw) if self.observe else self.raw
        if self.fail:
            raise RuntimeError("synthetic native failure")
        return [[SimpleNamespace(text=c, start_time=fixed[i*2]/1000, end_time=fixed[i*2+1]/1000)
                 for i,c in enumerate("你好啊")]]


def aligned():
    return capture_native_alignment(FakeNative(), None, "你好啊")


def six_aligned():
    return {"items":[{"text":c,"start_time":a,"end_time":b,"confidence":None}
                     for c,a,b in zip("你好啊世界吗",(.08,.4,.8,1.04,1.28,1.6),(.24,.56,.96,1.2,1.44,1.76))],
            "time_resolution_ms":80,
            "timestamp_audit":{"raw_timestamps_ms":[80,240,400,560,800,960,1040,1200,1280,1440,1600,1760],
                               "fixed_timestamps_ms":[80,240,400,560,800,960,1040,1200,1280,1440,1600,1760],
                               "repair_changed":False,"changes":[],"confidence":None}}


def record():
    return {"key": "C_demo/u.wav", "speaker": "C_demo", "audio": "/not-read-in-pure-tests.wav",
            "source_frames": 16000, "sample_rate": 16000, "segments": [[0, 16000]]}


class TimestampTests(unittest.TestCase):
    def test_native_raw_fixed_and_eighty_ms_not_one_ms(self):
        native = FakeNative(raw=[80, 240, 160, 560, 800, 960])
        r = capture_native_alignment(native, None, "你好啊")
        self.assertEqual(r["time_resolution_ms"], 80)
        self.assertEqual(r["timestamp_audit"]["raw_timestamps_ms"], [80,240,160,560,800,960])
        self.assertEqual(r["timestamp_audit"]["fixed_timestamps_ms"], [80,240,240,560,800,960])
        self.assertEqual(r["timestamp_audit"]["changes"], [{"index":2,"raw_ms":160,"fixed_ms":240,"delta_ms":80}])
        self.assertTrue(r["timestamp_audit"]["repair_changed"])
        self.assertIsNone(r["confidence"])
        self.assertFalse(r["boundary_accuracy_verified"])
        self.assertNotIn("fix_timestamp", vars(native.aligner_processor))
        json.dumps(r, allow_nan=False)

    def test_hook_restored_after_native_failure(self):
        native = FakeNative(fail=True)
        with self.assertRaises(AlignmentRefused) as caught:
            capture_native_alignment(native, None, "你好啊")
        self.assertEqual(len(caught.exception.evidence), 1)
        self.assertNotIn("fix_timestamp", vars(native.aligner_processor))

    def test_existing_instance_override_is_restored(self):
        native = FakeNative()
        override = lambda x: [int(i) for i in x]
        native.aligner_processor.fix_timestamp = override
        capture_native_alignment(native, None, "你好啊")
        self.assertIs(native.aligner_processor.fix_timestamp, override)

    def test_no_observed_raw_hook_is_refused(self):
        with self.assertRaisesRegex(AlignmentRefused, "observed_timestamp"):
            capture_native_alignment(FakeNative(observe=False), None, "你好啊")

    def test_han_character_rules_and_unknown_spoken_forms(self):
        self.assertEqual([c for c,_ in hanzi_text("你，好。啊")[1]], list("你好啊"))
        for text in ("2026", "你A好", "你好🙂", "!!!"):
            with self.subTest(text=text), self.assertRaises(AlignmentRefused):
                hanzi_text(text)

    def test_character_coverage_mismatch_is_not_filled(self):
        with self.assertRaisesRegex(AlignmentRefused, "coverage"):
            validate_character_alignment("你好", aligned()["items"], 16000, 16000)

    def test_zero_nonfinite_overlap_and_outside_are_refused(self):
        for start, end in ((.08,.08), (float("nan"),.24), (-.08,.24), (.08,1.08), (.4,.56)):
            r = aligned()["items"]
            r[0].update(start_time=start, end_time=end)
            with self.subTest(start=start,end=end), self.assertRaises(AlignmentRefused):
                validate_character_alignment("你好啊", r, 16000, 16000)


class CandidateTests(unittest.TestCase):
    def test_whole_character_counts_keep_internal_pause(self):
        rows = build_candidate_spans(record(), "你好啊", aligned())
        self.assertEqual([r["unit_count"] for r in rows], [1,2,3,1,2,1])
        two = rows[1]
        self.assertEqual(two["source_intervals"], [[1280,8960]])
        self.assertEqual(two["num_samples"], 7680)  # includes the 160ms gap
        self.assertEqual(two["text"], "你好")
        self.assertTrue(two["internal_pauses_retained"])
        self.assertFalse(two["vad_used_to_trim"])

    def test_context_only_uses_half_estimated_gap_not_adjacent_character(self):
        rows = build_candidate_spans(record(), "你好啊", aligned(), context_ms=40)
        second_char = next(r for r in rows if r["char_start_index"] == 1 and r["unit_count"] == 1)
        self.assertEqual((second_char["context_left_samples"],second_char["context_right_samples"]),(640,640))
        self.assertGreaterEqual(second_char["source_start_sample"], 3840)
        self.assertLessEqual(second_char["source_end_sample"], 12800)
        self.assertEqual(second_char["context_basis"], "estimated_alignment_gap_unverified")

    def test_vad_never_chops_weak_onset_or_tail(self):
        source = record(); source["segments"] = [[2000,3000]]
        row = build_candidate_spans(source, "你好啊", aligned(), max_units=1)[0]
        self.assertEqual((row["source_start_sample"],row["source_end_sample"]),(1280,3840))
        self.assertIn("low_predicted_vad_coverage_do_not_trim_target", row["review_reasons"])

    def test_every_candidate_stays_pending_even_monotone_and_full_vad(self):
        for row in build_candidate_spans(record(), "你好啊", aligned()):
            self.assertEqual(row["evidence"], "forced_alignment_candidate")
            self.assertEqual(row["boundary_review"], "pending")
            self.assertEqual(row["transcript_origin"], "asr_prediction")
            self.assertIsNone(row["confidence"])
            self.assertFalse(row["gold_eligible"])

    def test_repair_flag_does_not_disappear(self):
        a = capture_native_alignment(FakeNative(raw=[80,240,160,560,800,960]), None, "你好啊")
        row = build_candidate_spans(record(), "你好啊", a, max_units=1)[0]
        self.assertTrue(row["timestamp_repaired"])
        self.assertIn("native_timestamp_repair", row["review_reasons"])

    def test_budgets_and_vad_geometry_refused(self):
        for kwargs in ({"max_units":0},{"context_ms":41}):
            with self.assertRaises(AlignmentRefused):build_candidate_spans(record(),"你好啊",aligned(),**kwargs)
        for intervals in ([[0,16001]], [[100,200],[150,300]]):
            r = record();r["segments"] = intervals
            with self.assertRaises(AlignmentRefused):build_candidate_spans(r,"你好啊",aligned())

    def test_backend_requires_explicit_cpu_offline_before_model_import(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES":"0", "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1"}):
            with self.assertRaisesRegex(AlignmentRefused, "hidden_cuda"):
                NativeQwenCPUBackend("not-read", "not-read")

    def test_matched_family_exact_counts_same_anchor_and_no_partial_family(self):
        r = record();r["source_frames"] = 32000
        rows = matched_prefix_spans(r,"你好啊世界吗",six_aligned(),max_units=7,context_ms=20)
        self.assertEqual([x["unit_count"] for x in rows],[1,2,3,6])
        self.assertEqual(len({x["source_start_sample"] for x in rows}),1)
        self.assertEqual([x["text"] for x in rows],["你","你好","你好啊","你好啊世界吗"])
        self.assertTrue(all(a["source_end_sample"]<b["source_end_sample"] for a,b in zip(rows,rows[1:])))

    def test_incomplete_six_character_source_is_not_upgraded_to_family(self):
        with self.assertRaisesRegex(AlignmentRefused,"six_character"):
            matched_prefix_spans(record(),"你好啊",aligned())


@unittest.skipUnless(HAS_SOUNDFILE, "real file integration requires existing soundfile; no installation attempted")
class FileIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name);self.audio = self.root/"source.wav"
        with wave.open(str(self.audio), "wb") as out:
            out.setparams((1,2,16000,32000,"NONE","not compressed"))
            out.writeframes(struct.pack("<h",500)*32000)
        self.rec = dict(record(), audio=str(self.audio), source_frames=32000, corpus="C", split="train")
        class Backend:
            identity = {"synthetic_test_only":True}
            def transcribe(self,waveform,rate):return {"text":"你好啊世界吗","language":"Chinese","confidence":None}
            def align(self,waveform,rate,text):return six_aligned()
        self.backend = Backend()

    def test_public_schema_units_default_rejected_and_identity_sealed(self):
        from word_ft.schema import load_dataset, write_dataset
        r = align_training_record(self.rec,self.backend,max_units=6)
        self.assertEqual(len(r["units"]),4)
        self.assertEqual({x["source_utterance_id"] for x in r["units"]},{self.rec["key"]})
        self.assertEqual(len({x["family_id"] for x in r["units"]}),1)
        self.assertEqual([x["n_units"] for x in r["units"]],[1,2,3,6])
        self.assertEqual(r["units"][0]["vad"]["source_sha256"],r["units"][0]["audio_sha256"])
        self.assertEqual(r["units"][1]["n_samples_16k"],7680)
        audit = dict(r["audit"]);checksum = audit.pop("record_audit_sha256")
        self.assertEqual(identity_digest(audit),checksum)
        p = self.root/"units.json";write_dataset(p,r["units"],{"audit":r["audit"]})
        with self.assertRaisesRegex(ValueError,"candidate"):
            load_dataset(p)
        self.assertEqual(len(load_dataset(p,allow_candidates=True)["records"]),4)

    def test_long_source_refused_without_asr_or_time_cut(self):
        with patch.object(self.backend,"transcribe",side_effect=AssertionError("must not run")):
            with self.assertRaisesRegex(AlignmentRefused,"exceeds_budget"):
                align_training_record(self.rec,self.backend,max_audio_seconds=.5)

    def test_changed_source_during_alignment_refused(self):
        def mutate(*args):
            with self.audio.open("ab") as out:out.write(b"drift")
            return six_aligned()
        self.backend.align = mutate
        with self.assertRaisesRegex(AlignmentRefused,"source_content_changed|source_changed"):
            align_training_record(self.rec,self.backend,max_units=4)

    def test_bad_prediction_preserved_as_failure_evidence(self):
        self.backend.transcribe = lambda *args:{"text":"你好2026","language":"Chinese","confidence":None}
        with self.assertRaises(AlignmentRefused) as caught:
            align_training_record(self.rec,self.backend)
        self.assertEqual(caught.exception.evidence["asr_prediction"]["text"],"你好2026")

    def test_cli_dry_run_and_execute_use_single_json_candidates(self):
        path = Path(__file__).resolve().parents[1]/"scripts/align_training_units.py"
        spec = importlib.util.spec_from_file_location("align_training_cli_test",path)
        mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        source = self.root/"input.json";source.write_text(json.dumps({"records":[self.rec]}))
        out = self.root/"candidates"
        args = ["--input",str(source),"--output",str(out),"--asr-model","/not-read/asr","--aligner-model","/not-read/fa"]
        with contextlib.redirect_stdout(io.StringIO()),patch("word_ft.align.NativeQwenCPUBackend",side_effect=AssertionError("must not load")):
            self.assertEqual(mod.main(args),0)
        self.assertFalse(out.exists())
        with patch.dict(os.environ,dict(os.environ)),contextlib.redirect_stdout(io.StringIO()),patch("word_ft.align.NativeQwenCPUBackend",return_value=self.backend):
            self.assertEqual(mod.main(args+["--execute"]),0)
        obj = json.loads((out/"candidate_units.json").read_text())
        self.assertIsInstance(obj,dict)
        self.assertEqual(obj["schema"],"word-complete-units/v1")
        self.assertTrue(all(r["boundary_review"]=="pending" for r in obj["records"]))
        self.assertTrue((out/"human_review_queue.json").is_file())


if __name__ == "__main__":
    unittest.main()
