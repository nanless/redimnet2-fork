"""Offline, one-GPU ASR+FA replica; native tokens/times are never repaired here."""
from __future__ import annotations
import copy
import math
import time
import traceback
import unicodedata
from pathlib import Path
import sys
from contracts import native_json, need, sha_file, verify_model_stats, verify_models, validate_execution_policy
from normalization import normalize_with_audit
from alignment_observer import AlignmentObserver, successful_alignment, positive_integer_attribute


def normalize(text):
    return normalize_with_audit(text)["normalized_text"]


def length_batches(rows, max_batch, ratio):
    ordered = list(rows) if max_batch == 1 else sorted(rows, key=lambda r: (len(r["audio"]), r["row_id"]))
    batch = []
    for row in ordered:
        need(len(row["audio"]) > 0, "empty model input")
        if batch and (len(batch) >= max_batch or len(row["audio"]) / len(batch[0]["audio"]) > ratio):
            yield batch; batch = []
        batch.append(row)
    if batch: yield batch


class QwenBackend:
    name = "qwen3-asr-plus-ultra-no-head"
    def __init__(self, contract):
        validate_execution_policy(contract)
        need(contract.get("alignment", {}).get("mode") == "ultra_no_head", "production model must be explicit Ultra no-head")
        import torch
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
        self.torch, self.contract = torch, contract
        self.stats = {"asr_seconds": 0.0, "aligner_seconds": 0.0, "asr_rows": 0, "aligner_rows": 0,
                      "asr_batches": 0, "aligner_batches": 0, "asr_call_rows": {}, "aligner_call_rows": {}, "model_API_calls": 0}
        self.model_facts = verify_models(contract, full=False)
        need(torch.cuda.device_count() == 1, "production replica must see exactly one GPU")
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        runtime = contract["runtime"]
        started = time.monotonic()
        self.asr = Qwen3ASRModel.from_pretrained(contract["models"]["asr"]["path"], dtype=torch.float32,
                    device_map="cuda:0", local_files_only=True, trust_remote_code=False,
                    attn_implementation="eager", max_inference_batch_size=runtime["max_batch"],
                    max_new_tokens=runtime["max_new_tokens"])
        self.aligner = Qwen3ForcedAligner.from_pretrained(contract["models"]["aligner"]["path"], dtype=torch.float32,
                    device_map="cuda:0", local_files_only=True, trust_remote_code=False, attn_implementation="eager")
        for model in (self.asr, self.aligner): model.model.eval()
        need(all(model.model.training is False for model in (self.asr, self.aligner)), "loaded models are not in eval mode")
        self.timestamp_attributes = {"timestamp_token_id": positive_integer_attribute(self.aligner.timestamp_token_id, "timestamp_token_id"),
                                     "timestamp_segment_ms": positive_integer_attribute(self.aligner.timestamp_segment_time, "timestamp_segment_time")}
        self.asr.model.generation_config.do_sample = False
        self.asr.model.generation_config.temperature = None; self.asr.model.generation_config.top_p = None
        self.stats["model_load_seconds"] = time.monotonic() - started
        self._captures = []
        original = self.asr.model.generate
        def capture_generate(*args, **kwargs):
            observed_ids = kwargs.get("input_ids")
            need(observed_ids is not None and observed_ids.ndim == 2 and observed_ids.shape[0] == 1,
                 "actual ASR generate must be exactly one row")
            need(self.asr.model.training is False and torch.is_inference_mode_enabled() and not torch.is_grad_enabled(),
                 "actual ASR not eval/inference-only")
            result = original(*args, **kwargs)
            seq = result.sequences if hasattr(result, "sequences") else result
            input_ids = kwargs.get("input_ids"); need(input_ids is not None, "native ASR input tokens unobserved")
            generated = seq[:, input_ids.shape[1]:]
            rows = generated.detach().cpu().tolist()
            text = self.asr.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            eos = self.asr.model.generation_config.eos_token_id
            eos = [eos] if type(eos) is int else list(eos or [])
            self._captures.extend({"generated_token_ids": ids, "native_decoded_raw_answer": raw,
                                   "eos_observed": any(x in eos for x in ids),
                                   "first_eos_index": next((j for j, x in enumerate(ids) if x in eos), None),
                                   "max_new_tokens": kwargs.get("max_new_tokens"),
                                   "input_padded_tokens": int(input_ids.shape[1]), "actual_generate_rows": int(input_ids.shape[0]),
                                   "model_training": self.asr.model.training, "inference_mode": bool(torch.is_inference_mode_enabled()),
                                   "grad_enabled": bool(torch.is_grad_enabled())} for ids, raw in zip(rows, text))
            return result
        self.asr.model.generate = capture_generate
        languages = self.aligner.get_supported_languages()
        self.supported_languages = {str(x).lower() for x in languages} if languages is not None else None
        self.module_provenance = self.verify_loaded_modules()

    def verify_loaded_modules(self):
        expected = {str(Path(path).resolve()): pin for path, pin in self.contract["source_files"].items()
                    if "qwen_asr" in Path(path).parts and Path(path).suffix == ".py"}
        required = {"qwen_asr", "qwen_asr.inference.qwen3_asr", "qwen_asr.inference.qwen3_forced_aligner", "qwen_asr.inference.utils"}
        need(required <= set(sys.modules), "required Qwen modules were not actually imported")
        result = {}
        for name, module in list(sys.modules.items()):
            if not (name == "qwen_asr" or name.startswith("qwen_asr.")): continue
            raw = getattr(module, "__file__", None)
            if raw is None: continue
            path = str(Path(raw).resolve())
            need(path in expected and sha_file(path) == expected[path], "actually imported Qwen module is not the pinned source: " + name + " " + path)
            result[name] = {"path": path, "sha256": expected[path]}
        return result

    def _observe_call(self, stage, batch):
        policy = validate_execution_policy(self.contract)
        if policy == "single_window_independent":
            need(len(batch) == 1, "single-window contract reached a multirow " + stage + " model call")
        widths = self.stats[stage + "_call_rows"]; key = str(len(batch))
        widths[key] = widths.get(key, 0) + 1
        return {"stage": stage, "rows": len(batch), "producer_row_ids": [r["row_id"] for r in batch],
                "input_frames": [len(r["audio"]) for r in batch], "execution_policy": policy}

    def transcribe(self, rows):
        torch = self.torch; runtime = self.contract["runtime"]; outputs = {}
        for batch in length_batches(rows, runtime["max_batch"], runtime["length_ratio_limit"]):
            self._captures.clear(); started = time.monotonic(); ids = [r["row_id"] for r in batch]
            call = self._observe_call("asr", batch)
            try:
                torch.cuda.synchronize(0)
                with torch.inference_mode():
                    result = self.asr.transcribe(audio=[(r["audio"], 16000) for r in batch], language=None, return_time_stamps=False)
                torch.cuda.synchronize(0)
                need(len(result) == len(batch) == len(self._captures), "ASR native row coverage differs")
                for row_id, answer, native in zip(ids, result, self._captures):
                    outputs[row_id] = {"status": "completed" if native["eos_observed"] else "truncated", "raw_text": str(answer.text),
                                       "normalized_text": normalize(str(answer.text)), "normalization_audit": normalize_with_audit(str(answer.text)), "language": answer.language,
                                       "native": native, "batch_rows": ids, "batch_size": len(batch), "inference_call": call}
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA" in str(exc) or "device-side assert" in str(exc):
                    raise RuntimeError("fatal GPU error; preserve attempt, no silent batch reduction") from exc
                trace = traceback.format_exc()
                for row_id in ids: outputs[row_id] = {"status": "failed", "raw_text": "", "normalized_text": "", "language": None,
                                                       "native": {"eos_observed": False}, "normalization_audit": normalize_with_audit(""), "error": trace,
                                                       "native_generate_captures": list(self._captures), "batch_rows": ids, "inference_call": call}
            finally:
                self.stats["asr_seconds"] += time.monotonic() - started; self.stats["asr_rows"] += len(batch); self.stats["asr_batches"] += 1
        return native_json(outputs)

    def align(self, rows):
        torch = self.torch; runtime = self.contract["runtime"]; outputs = {}; eligible = []
        for row in rows:
            asr = row["asr"]; row_id = row["row_id"]
            reason = None
            if asr["status"] != "completed": reason = "asr_not_complete"
            elif not asr["normalized_text"]: reason = "empty_asr_not_proven_noise"
            elif self.supported_languages is not None and str(asr["language"]).lower() not in self.supported_languages: reason = "unsupported_alignment_language"
            if reason:
                outputs[row_id] = {"status": "not_applicable", "reason": reason, "items": [], "target_text": asr.get("normalized_text", ""),
                                   "native_timestamps_ms": [], "fixed_timestamps_ms": [], "repair_changed": False}
            else: eligible.append(row)
        for batch in length_batches(eligible, runtime["max_batch"], runtime["length_ratio_limit"]):
            need(len(batch) == 1, "Ultra no-head must execute exactly one window per call")
            row = batch[0]; row_id = row["row_id"]; started = time.monotonic(); observer = None
            call = self._observe_call("aligner", batch)
            try:
                observer = AlignmentObserver(self.aligner, torch)
                torch.cuda.synchronize(0)
                with torch.inference_mode(), observer:
                    result = self.aligner.align(audio=[(row["audio"], 16000)],
                        text=[row["asr"]["normalized_text"]], language=[row["asr"]["language"]])
                torch.cuda.synchronize(0)
                need(len(result) == 1, "Ultra no-head output row coverage differs")
                items = [{"text": str(x.text), "start_time": float(x.start_time), "end_time": float(x.end_time)} for x in result[0]]
                outputs[row_id] = successful_alignment(observer, items, row["asr"]["normalized_text"])
                outputs[row_id].update(batch_rows=[row_id], inference_call=call,
                                      actual_supported_languages=sorted(self.supported_languages) if self.supported_languages is not None else None)
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA" in str(exc) or "device-side assert" in str(exc):
                    raise RuntimeError("fatal GPU error; preserve attempt, no silent batch reduction") from exc
                observation = observer.observation(failed=True) if observer is not None else {"status": "observer_initialization_failed"}
                forward = observation.get("thinker_forwards", [])
                fix = observation.get("FIX_calls", [])
                outputs[row_id] = {"status": "failed", "method": "ultra_no_head", "items": [],
                    "native_timestamps_ms": forward[0].get("raw_timestamps_ms", []) if forward else [],
                    "fixed_timestamps_ms": fix[-1].get("fixed_timestamps_ms") if fix else None,
                    "repair_changed": fix[-1].get("repair_changed") if fix else None,
                    "target_text": row["asr"]["normalized_text"], "error": traceback.format_exc(),
                    "native_observation": observation, "batch_rows": [row_id], "inference_call": call,
                    "head_used": False, "extra_clipping_or_interpolation": False}
            finally:
                self.stats["aligner_seconds"] += time.monotonic() - started
                self.stats["aligner_rows"] += 1; self.stats["aligner_batches"] += 1

        return native_json(outputs)

    def metrics(self):
        torch = self.torch; torch.cuda.synchronize(0)
        if validate_execution_policy(self.contract) == "single_window_independent":
            for stage in ("asr", "aligner"):
                need(set(self.stats[stage + "_call_rows"]) <= {"1"}
                     and self.stats[stage + "_rows"] == self.stats[stage + "_batches"],
                     "observed model calls violated single-window semantics: " + stage)
        verify_model_stats(self.model_facts)
        actual_modules = self.verify_loaded_modules()
        need(all(actual_modules.get(name) == value for name, value in self.module_provenance.items()), "runtime Qwen module bytes changed")
        need(all(model.model.training is False for model in (self.asr, self.aligner)), "model eval state changed")
        return self.stats | {"alignment_mode": "ultra_no_head", "timestamp_attributes": self.timestamp_attributes,
                             "actual_model_training_states": {"asr": self.asr.model.training, "aligner": self.aligner.model.training},
                             "cuda_allocated_peak_bytes": torch.cuda.max_memory_allocated(0),
                             "cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved(0)}


class FixtureBackend:
    """Explicit synthetic CPU fixtures; prohibited for silver/pilot contracts."""
    name = "synthetic_cpu_fixture"
    def __init__(self, contract):
        need(contract["purpose"] == "cpu_fixture", "synthetic backend cannot label real production")
        self.contract = contract
    def transcribe(self, rows):
        return {r["row_id"]: {"status": "completed", "raw_text": "fixture", "normalized_text": "fixture", "normalization_audit": normalize_with_audit("fixture"), "language": "English",
                             "native": {"eos_observed": True, "generated_token_ids": [1, 2], "native_decoded_raw_answer": "fixture"}} for r in rows}
    def align(self, rows):
        return {r["row_id"]: {"status": "completed_native_candidates", "items": [{"text": "fixture", "start_time": 0.08, "end_time": 0.16}],
                             "native_timestamps_ms": [80, 160], "fixed_timestamps_ms": [80, 160], "repair_changed": False,
                             "target_text": r["asr"]["normalized_text"]} for r in rows}
    def metrics(self): return {"model_API_calls": 0, "synthetic_fixture": True}
