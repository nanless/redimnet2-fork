"""Synthetic adversarial contract tests; no audio/model/API/GPU calls."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import schema as S


def fixture():
    dataset, relative = "fixture", "speaker/a.wav"
    source = {"schema": S.SOURCE_SCHEMA, "source_id": "src_" + hashlib.sha256((dataset + "\0" + relative).encode()).hexdigest(),
              "dataset": dataset, "version": "synthetic-v1", "relative_path": relative,
              "audio_path": "/synthetic/never_read/a.wav", "resolution_status": "resolved",
              "sidecar_relpath": dataset + "/" + relative + ".annotation.json",
              "roles": ["train_ptn", "train_lm"], "original_splits": ["train"], "parent_id": "synthetic-parent",
              "protection": {"training_forbidden": False, "reasons": []},
              "channel_policy": "mean_channels_to_mono", "source_interval": None,
              "speaker_evidence": {"provider_label": "synthetic-speaker"}}
    identity = {"decode_status": "decoded_complete", "file_sha256": "1"*64, "pcm_sha256": "2"*64,
                "pcm_hash_policy": S.PCM_HASH_POLICY, "bytes": 350000, "sample_rate": 44100,
                "channels": 2, "frames": 88200, "duration_seconds": 2.0,
                "stat_before": [1, 22, 350000, 123456789], "stat_after": [1, 22, 350000, 123456789]}
    contract = {"sha256": "3"*64, "version": "fixture-v1", "api_calls": 0,
                "provenance": {"models": {"asr": {"path": "/synthetic/asr", "files": {"weights": {"bytes":100,"sha256":"4"*64}}},
                                          "aligner": {"path": "/synthetic/aligner", "files": {"weights": {"bytes":100,"sha256":"5"*64}}}},
                               "code_sha256": "6"*64}}
    for model in contract["provenance"]["models"].values():model["files_sha256"]=S.sha256_json(model["files"])
    vad = {"status": "not_available", "reused": False, "segments": [], "errors": []}
    window = {"window_id": "w000", "context_start_sample": 0, "context_end_sample": 88200,
              "owned_start_sample": 0, "owned_end_sample": 88200, "model_sample_rate": 16000,
              "time_mapping": S.TIME_MAPPING, "input_pcm_sha256": "7"*64,
              "asr": {"status": "completed", "raw_text": "哈哈", "normalized_text": "哈哈", "language": "Chinese",
                      "normalization_audit": S.normalize_with_audit("哈哈"),
                      "native": {"eos_observed": True, "generated_token_ids": [1, 2, 3], "max_new_tokens": 12}},
              "alignment": {"status": "completed_native_candidates", "target_text": "哈哈", "items": [
                  {"text": "哈", "start_time": .1, "end_time": .4},
                  {"text": "哈", "start_time": .8, "end_time": 1.2}],
                  "native_timestamps_ms": [100, 400, 800, 1200], "fixed_timestamps_ms": [100, 400, 800, 1200], "repair_changed": False}}
    units = [{"text": "哈", "start_sample": math.floor(a*44100), "end_sample": math.ceil(b*44100),
              "sample_rate": 44100, "window_id": "w000", "native_index": i, "occurrence_index": i,
              "geometry_status": "valid", "repair_applied": False, "boundary_status": "machine_candidate"}
             for i, (a,b) in enumerate(((.1,.4),(.8,1.2)))]
    execution = {"job_key": "job-fixture", "attempt_id": "attempt-fixture", "worker_id": 0,
                 "device": "synthetic_no_device", "started_utc": "2026-10-24T00:00:00Z",
                 "finished_utc": "2026-10-24T00:00:01Z", "api_calls": 0}
    return source, identity, contract, vad, [window], units, execution, []


class QualitySchemaTests(unittest.TestCase):
    def record(self): return S.build_record(*fixture())

    def test_valid_candidate_and_canonical_stability(self):
        r=self.record(); self.assertEqual(r["decision"]["state"],"candidate"); self.assertEqual(S.validate_record(r),[])
        self.assertEqual(S.canonical_bytes(r),S.canonical_bytes(json.loads(S.canonical_bytes(r))))
        self.assertFalse(r["decision"]["usable_for_training"]); self.assertIsNone(r["decision"]["human_review"])

    def test_builder_does_not_mutate_source_observations(self):
        args=fixture();before=deepcopy(args);r=S.build_record(*args);r["source"]["roles"].append("x")
        self.assertEqual(args,before)

    def test_native_zero_cannot_be_hidden_by_filtering_top_units(self):
        args=list(fixture());args[4][0]["alignment"]["items"][1]["end_time"]=.8
        args[4][0]["alignment"]["native_timestamps_ms"][-1]=800;args[4][0]["alignment"]["fixed_timestamps_ms"][-1]=800
        args[5]=args[5][:1];r=S.build_record(*args)
        self.assertEqual(r["decision"]["state"],"unknown"); self.assertEqual(S.validate_record(r),[])
        r["decision"]["state"]="candidate"; self.assertTrue(any("candidate_hides" in x for x in S.validate_record(r)))

    def test_repair_not_upgraded_to_clean_candidate(self):
        args=list(fixture());args[4][0]["alignment"]["repair_changed"]=True
        for u in args[5]:u["repair_applied"]=True
        r=S.build_record(*args); self.assertEqual(r["decision"]["state"],"unknown");self.assertEqual(S.validate_record(r),[])

    def test_empty_asr_is_unknown_not_noise(self):
        args=list(fixture());args[4][0]["asr"]["raw_text"]=""
        r=S.build_record(*args);self.assertEqual(r["decision"]["state"],"unknown")
        self.assertTrue(any("empty_asr_not_proven_noise" in x for x in r["decision"]["reasons"]))

    def test_unsupported_is_terminal_unknown(self):
        args=list(fixture());args[4][0]["alignment"]["status"]="unsupported"
        r=S.build_record(*args);self.assertEqual(r["decision"]["state"],"unknown");self.assertTrue(r["decision"]["annotation_completed"])

    def test_truncated_asr_eos_not_silently_succeeded(self):
        args=list(fixture());args[4][0]["asr"]["native"]["eos_observed"]=False
        self.assertEqual(S.build_record(*args)["decision"]["state"],"unknown")

    def test_native_to_source_clock_uses_original_44100_rate(self):
        r=self.record();r["units"][0]["sample_rate"]=16000;r["units"][0]["start_sample"]=1600
        e=S.validate_record(r);self.assertIn("unit:wrong_source_sample_rate",e);self.assertIn("unit:source_clock_mapping_mismatch",e)

    def test_false_repair_flag_detected(self):
        args=list(fixture());args[4][0]["alignment"]["fixed_timestamps_ms"][0]=80
        r=S.build_record(*args);self.assertEqual(r["decision"]["state"],"unknown")
        self.assertTrue(any("repair_flag_contradiction" in x for x in r["decision"]["reasons"]))

    def test_machine_review_cannot_be_forged_human_or_release(self):
        r=self.record();r["decision"]["human_review"]={"reviewer":"a model"};r["decision"]["usable_for_training"]=True
        self.assertIn("decision:machine_output_claims_human_review",S.validate_record(r))
        self.assertIn("decision:unreviewed_consumer_release",S.validate_record(r))

    def test_path_traversal_rejected(self):
        r=self.record();r["source"]["relative_path"]="speaker/../secret.wav"
        self.assertIn("source:unsafe_relative_path",S.validate_record(r))

    def test_extension_preserved_prevents_sidecar_collision(self):
        a=self.record()["source"];b=deepcopy(a);b["relative_path"]="speaker/a.flac";b["sidecar_relpath"]="fixture/speaker/a.flac.annotation.json"
        b["source_id"]="src_"+hashlib.sha256((b["dataset"]+"\0"+b["relative_path"]).encode()).hexdigest()
        self.assertNotEqual(a["source_id"],b["source_id"]);self.assertNotEqual(a["sidecar_relpath"],b["sidecar_relpath"])
        r=self.record();r["source"]=b;self.assertEqual(S.validate_record(r),[])

    def test_repeated_text_occurrences_not_dictionary_collapsed(self):
        r=self.record();r["units"][1]["occurrence_index"]=0
        self.assertIn("unit:repeated_text_occurrence_mismatch",S.validate_record(r))

    def test_unresolved_source_remains_terminal_denominator(self):
        args=list(fixture());args[0]["audio_path"]=None;args[0]["resolution_status"]="unresolved"
        args[1]={"decode_status":"missing","file_sha256":None,"pcm_sha256":None,"sample_rate":None,"channels":None,"frames":None,"bytes":None,"duration_seconds":None,"stat_before":None,"stat_after":None};args[4]=[];args[5]=[]
        r=S.build_record(*args);self.assertEqual(r["decision"]["state"],"unknown");self.assertEqual(S.validate_record(r),[])

    def test_protected_source_does_not_gain_training_permission(self):
        args=list(fixture());args[0]["protection"]={"training_forbidden":True,"reasons":["diagnostic"]}
        r=S.build_record(*args);self.assertFalse(r["decision"]["usable_for_training"]);self.assertEqual(S.validate_record(r),[])

    def test_duration_clock_contradiction_rejected(self):
        r=self.record();r["source_identity"]["duration_seconds"]=3.
        self.assertIn("identity:duration_frames_clock_mismatch",S.validate_record(r))

    def test_model_api_calls_rejected(self):
        r=self.record();r["execution"]["api_calls"]=1;r["contract"]["api_calls"]=True
        self.assertIn("execution:production_api_not_zero",S.validate_record(r));self.assertIn("contract:production_api_not_zero",S.validate_record(r))

    def test_native_nonfinite_rejected_as_json_but_explicit_encoding_is_unknown(self):
        r=self.record();r["windows"][0]["alignment"]["native_timestamps_ms"][0]=float("nan")
        self.assertTrue(any("nonfinite_json" in x for x in S.validate_record(r)))
        with self.assertRaises(ValueError):S.canonical_bytes(r)
        args=list(fixture());args[4][0]["alignment"]["native_timestamps_ms"][0]={"invalid_native_float":"nan"}
        q=S.build_record(*args);self.assertEqual(q["decision"]["state"],"unknown");self.assertEqual(S.validate_record(q),[])

    def test_owned_window_coverage_gap_is_unknown(self):
        args=list(fixture());args[4][0]["owned_end_sample"]=70000
        r=S.build_record(*args);self.assertIn("owned_windows_do_not_partition_complete_source",r["decision"]["reasons"])

    def test_unit_cannot_be_assigned_to_wrong_owned_window(self):
        r=self.record();r["windows"][0]["owned_start_sample"]=30000
        self.assertIn("unit:wrong_owned_window",S.validate_record(r))

    def test_bad_native_index_rejected(self):
        r=self.record();r["units"][0]["native_index"]=20
        self.assertIn("unit:native_index_outside_items",S.validate_record(r))

    def test_source_id_binds_path_not_claimed_pcm_dedup(self):
        r=self.record();r["source"]["source_id"]="src_"+"f"*64
        self.assertIn("source:id_not_bound_to_dataset_relative_path",S.validate_record(r))

    def test_malformed_record_returns_reasons_not_unhandled_exception(self):
        for r in (None,[],{}, {"schema":S.SCHEMA,"source":None}):self.assertTrue(S.validate_record(r))

    def test_zero_frame_source_remains_unknown_terminal(self):
        args=list(fixture());args[1]["frames"]=0;args[1]["duration_seconds"]=0.;args[4]=[];args[5]=[]
        r=S.build_record(*args);self.assertEqual(r["decision"]["state"],"unknown");self.assertEqual(S.validate_record(r),[])

    def test_pinned_model_digest_cannot_be_replaced_by_path_only(self):
        r=self.record();r["provenance"]["models"]["asr"]["files"]["weights"]["sha256"]="8"*64
        self.assertIn("provenance:asr_file_pin_digest_mismatch",S.validate_record(r))

    def test_legacy_vad_cannot_gain_fabricated_historical_source_sha(self):
        r=self.record();r["source"]["old_vad"]={"identity_binding":"legacy_source_path_size_mtime_clock"}
        r["vad"]={"status":"verified_metadata","reused":True,"segments":[],"original_path":"/synthetic/a.rttm","original_sha256":"a"*64,
                  "metadata_path":"/synthetic/a.meta.json","metadata_sha256":"b"*64,"identity_binding":"legacy_source_path_size_mtime_clock"}
        self.assertEqual(S.validate_record(r),[])
        r["vad"]["old_source_sha256"]=r["source_identity"]["file_sha256"]
        self.assertIn("vad:legacy_binding_upgraded_to_historical_source_sha",S.validate_record(r))

    def test_alignment_wrong_target_cannot_pass_on_geometry_alone(self):
        args=list(fixture());args[4][0]["alignment"]["target_text"]="乙乙"
        r=S.build_record(*args);self.assertEqual(r["decision"]["state"],"unknown")
        self.assertTrue(any("alignment_target_not_bound" in x for x in r["decision"]["reasons"]))

    def test_missing_native_text_occurrence_is_unknown(self):
        args=list(fixture());args[4][0]["asr"]["normalized_text"]="哈哈哈";args[4][0]["alignment"]["target_text"]="哈哈哈"
        r=S.build_record(*args);self.assertEqual(r["decision"]["state"],"unknown")
        self.assertTrue(any("text_coverage_mismatch" in x for x in r["decision"]["reasons"]))

    def test_failed_window_preprocessing_has_no_invented_pcm_digest(self):
        args=list(fixture());w=args[4][0];w["preprocessing_status"]="failed";w["input_pcm_sha256"]=None
        w["asr"]={"status":"failed","raw_text":"","normalized_text":"","native":{"eos_observed":False},"error":"actual preserved seek failure"}
        w["alignment"]={"status":"not_applicable","target_text":"","items":[],"native_timestamps_ms":[],"fixed_timestamps_ms":[],"repair_changed":False};args[5]=[]
        r=S.build_record(*args);self.assertEqual(r["decision"]["state"],"unknown");self.assertEqual(S.validate_record(r),[])
        r["windows"][0]["preprocessing_status"]="completed"
        self.assertIn("window:model_pcm_identity_missing",S.validate_record(r))

    def test_spoken_form_uncertainty_cannot_pass_on_valid_geometry(self):
        for raw in ("123", "３.１４", "I'm", "API", "你好world", "a\u0301"):
            with self.subTest(raw=raw):
                args=list(fixture());asr=args[4][0]["asr"];audit=S.normalize_with_audit(raw)
                asr.update(raw_text=raw,normalized_text=audit["normalized_text"],normalization_audit=audit)
                # Native aligner observations remain intact. Review marks the source;
                # it must not throw away ASR/native intervals to manufacture success.
                r=S.build_record(*args)
                self.assertEqual(r["decision"]["state"],"unknown")
                self.assertIn("window_0:spoken_form_review_required",r["decision"]["reasons"])
                self.assertEqual(S.validate_record(r),[])
                self.assertEqual(r["windows"][0]["asr"]["raw_text"],raw)
                self.assertEqual(len(r["windows"][0]["alignment"]["items"]),2)

    def test_false_normalization_audit_flag_and_raw_mapping_rejected(self):
        args=list(fixture());asr=args[4][0]["asr"];audit=S.normalize_with_audit("1")
        asr.update(raw_text="1",normalized_text="1",normalization_audit=audit)
        r=S.build_record(*args);r["windows"][0]["asr"]["normalization_audit"]["spoken_form_review_required"]=False
        self.assertIn("window:normalization_audit_not_reconstructible",S.validate_record(r))
        r=self.record();r["windows"][0]["asr"]["normalization_audit"]["normalized_character_map"][1]["raw_start"]=0
        self.assertIn("window:normalization_audit_not_reconstructible",S.validate_record(r))

    def test_plain_english_and_exact_nfkc_map_not_word_count_claim(self):
        for raw in ("hello world", "ﬃ 哈哈"):
            audit=S.normalize_with_audit(raw)
            if raw=="hello world":
                self.assertEqual(audit["normalized_text"],"hello world")
                self.assertFalse(audit["spoken_form_review_required"])
                self.assertEqual(len(audit["normalized_character_map"]),11)
            if raw.startswith("ﬃ"):
                self.assertEqual(audit["normalized_text"],"ffi 哈哈")
                self.assertEqual([(x["raw_start"],x["raw_end"]) for x in audit["normalized_character_map"][:3]],[(0,1)]*3)

    def test_clean_candidate_cannot_omit_intermediate_normalization(self):
        r=self.record();del r["windows"][0]["asr"]["normalization_audit"]["nfkc_text"]
        self.assertIn("window:normalization_audit_not_reconstructible",S.validate_record(r))

    def test_real_spaced_initialism_and_punctuation_variants_are_unknown(self):
        for raw in ("M V", "M\t\n V", "M,V", "M / V", "M.V.", "U S A", "M M", "ＡＰＩ"):
            with self.subTest(raw=raw):
                args=list(fixture());asr=args[4][0]["asr"];audit=S.normalize_with_audit(raw)
                asr.update(raw_text=raw,normalized_text=audit["normalized_text"],normalization_audit=audit)
                r=S.build_record(*args)
                self.assertTrue(audit["spoken_form_review_required"])
                self.assertEqual(r["decision"]["state"],"unknown")
                self.assertEqual(r["windows"][0]["alignment"]["native_timestamps_ms"],[100,400,800,1200])
                self.assertEqual(S.validate_record(r),[])

    def test_plain_repeated_words_or_uppercase_sentence_not_initialism_rule(self):
        for raw in ("hello hello", "Bag, bag.", "BAG BAG", "GO GO", "HELLO WORLD", "I am here", "I I", "A A"):
            with self.subTest(raw=raw):
                audit=S.normalize_with_audit(raw)
                self.assertFalse(audit["spoken_form_review_required"])
                self.assertEqual(audit["separated_uppercase_letter_runs"],[])
        self.assertEqual(S.normalize_with_audit("M V")["separated_uppercase_letter_runs"],[["M","V"]])

    def test_spaced_initialism_cannot_forge_false_review_or_erase_run(self):
        args=list(fixture());asr=args[4][0]["asr"];audit=S.normalize_with_audit("M V")
        asr.update(raw_text="M V",normalized_text="M V",normalization_audit=audit)
        for key,value in (("spoken_form_review_required",False),("separated_uppercase_letter_runs",[])):
            r=S.build_record(*args);r["windows"][0]["asr"]["normalization_audit"][key]=value
            self.assertIn("window:normalization_audit_not_reconstructible",S.validate_record(r))


if __name__ == "__main__":unittest.main()
