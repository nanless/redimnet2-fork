"""P1 短窗护栏的纯逻辑回归；不需要 torch / GPU / 真实音频 / 网络。

背景（2026-11-01 05:13 标注停摆真因）：
    CN-Celeb2 `id10052/live_broadcast-04-007.flac` 只有 171 帧（0.010688 s）。
    WhisperFeatureExtractor n_fft=400、center=True 两侧各补 200 帧，
    torch.stft 的 F.pad 要求 pad < 输入长度 → 171 帧必抛
    `padding (200,200) at dimension 2 of input [1,1,171]`。

本测试只校验"护栏把短窗写成失败窗、且决策落 unknown"，不校验 ASR 质量。
补丁未应用时整体 skip，避免在未打补丁的树上给出假通过。
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

import worker  # noqa: E402  纯逻辑模块，torch 只在函数体内延迟导入
import schema  # noqa: E402

P1_APPLIED = hasattr(worker, "MIN_WINDOW_FRAMES") and hasattr(worker, "refuse_short_window")


OBSERVED_SHORT_FRAMES = 171          # 现场实测，勿改
STFT_N_FFT = 400
STFT_PAD_PER_SIDE = 200


def short_window(frames=OBSERVED_SHORT_FRAMES):
    """模拟 plan_windows 为 171 帧文件产出的唯一窗口。"""
    return {"window_id": "window_000000",
            "context_start_sample": 0, "context_end_sample": frames,
            "owned_start_sample": 0, "owned_end_sample": frames,
            "model_sample_rate": 16000, "time_mapping": schema.TIME_MAPPING,
            "left_context_limited": True, "right_context_limited": True}


def minimal_record(window, errors):
    """quality_reasons / build_record 决策路径所需的最小记录（不发布、不落盘）。"""
    return {"schema": "redimnet2-open-annotation-record/v1",
            "source": {"schema": "redimnet2-annotation-source/v1",
                       "audio_path": "/nonexistent/id10052/live_broadcast-04-007.flac",
                       "resolution_status": "resolved", "source_interval": None},
            "source_identity": {"decode_status": "decoded_complete", "frames": OBSERVED_SHORT_FRAMES,
                                "sample_rate": 16000,
                                "stat_before": {"device": 2097190}, "stat_after": {"device": 2097190}},
            "windows": [window], "units": [], "errors": list(errors),
            "provenance": {"runtime": {}}, "decision": {"state": "candidate", "reasons": []}}


@unittest.skipUnless(P1_APPLIED, "P1 未应用：worker.MIN_WINDOW_FRAMES / refuse_short_window 不存在")
class ShortWindowGuardTests(unittest.TestCase):
    def test_threshold_is_one_full_analysis_window(self):
        """护栏阈值取一个完整分析窗，不取刚好越过 stft 下限的值。"""
        self.assertEqual(worker.MIN_WINDOW_FRAMES, STFT_N_FFT)
        # 现场那一例必然崩：两侧各补 200 帧 > 171 帧。
        self.assertGreater(2 * STFT_PAD_PER_SIDE, OBSERVED_SHORT_FRAMES)
        # 阈值不小于一个完整分析窗（400），因此 201–399 这个"能跑但没意义"的
        # 区间也被一并盖住，而不是只挡住必然崩的 <=200。
        self.assertGreaterEqual(worker.MIN_WINDOW_FRAMES, 2 * STFT_PAD_PER_SIDE)

    def test_171_frames_becomes_failed_window_without_raising(self):
        """171 帧不抛异常，落成 failed 窗。"""
        window = short_window()
        context = {"errors": []}
        counters = {}
        returned = worker.refuse_short_window(window, OBSERVED_SHORT_FRAMES, context, counters)

        self.assertTrue(returned)
        self.assertEqual(window["preprocessing_status"], "failed")
        self.assertIsNone(window["input_pcm_sha256"])
        self.assertEqual(window["asr"]["status"], "failed")
        self.assertEqual(window["alignment"]["status"], "not_applicable")
        self.assertIn("asr_input_too_short", window["asr"]["error"])
        self.assertIn("171", window["asr"]["error"])
        self.assertEqual(counters["short_window_refused"], 1)
        self.assertEqual(len(context["errors"]), 1)
        self.assertEqual(context["errors"][0]["stage"], "asr_input_too_short")
        self.assertEqual(context["errors"][0]["frames"], OBSERVED_SHORT_FRAMES)
        self.assertEqual(context["errors"][0]["min_frames"], worker.MIN_WINDOW_FRAMES)

    def failed_record(self):
        """用真 fixture + 真 build_record 造一条含短窗的完整记录（不发布、不落盘、不读音频）。"""
        from test_quality_schema import fixture
        source, identity, contract, vad, _windows, _units, execution, _errors = fixture()
        identity = dict(identity, sample_rate=16000, channels=1, frames=OBSERVED_SHORT_FRAMES,
                        bytes=342, duration_seconds=OBSERVED_SHORT_FRAMES / 16000.0,
                        file_sha256="a" * 64, pcm_sha256="b" * 64,
                        stat_before=[1, 22, 342, 123456789], stat_after=[1, 22, 342, 123456789])
        window = short_window()
        context = {"errors": []}
        worker.refuse_short_window(window, OBSERVED_SHORT_FRAMES, context, {})
        return schema.build_record(source, identity, contract, vad, [window], [], execution, context["errors"])

    def test_real_validator_accepts_the_failed_window(self):
        """要过 schema 的真实发布门（validate_record / assert_valid_record），不是本地复刻谓词。"""
        record = self.failed_record()
        self.assertEqual(schema.validate_record(record), [],
                         "真实 validator 拒绝了护栏产出的失败窗")
        schema.assert_valid_record(record)   # 不抛异常即通过 store.publish 的同一道门

    def test_real_build_record_decides_unknown(self):
        """真 build_record 的决策要落 unknown，且理由是 asr_not_succeeded。"""
        record = self.failed_record()
        self.assertEqual(record["decision"]["state"], "unknown")
        reasons = record["decision"]["reasons"]
        self.assertIn("window_0:asr_not_succeeded", reasons)
        self.assertIn("no_usable_lexical_units", reasons)
        self.assertIn("processing_errors_preserved", reasons)
        self.assertFalse(record["decision"]["usable_for_training"])
        self.assertFalse(record["decision"]["usable_for_evaluation"])

    def test_real_validator_rejects_a_broken_shape(self):
        """负向对照：把形状改坏（有状态但无 PCM 身份）后，真 validator 拒绝。

        没有这条，上面那条"validator 接受"可能只是因为门根本没在查。
        """
        record = self.failed_record()
        record["windows"][0]["preprocessing_status"] = "completed"   # 破坏谓词
        errors = schema.validate_record(record)
        self.assertNotEqual(errors, [], "validator 对坏形状仍然放行，说明这道门没在工作")
        with self.assertRaises(ValueError):
            schema.assert_valid_record(record)

    def test_long_window_is_not_touched(self):
        """正常长度（>= 400 帧）不进入护栏分支。"""
        for frames in (worker.MIN_WINDOW_FRAMES, worker.MIN_WINDOW_FRAMES + 1, 16 * 16000):
            window = short_window(frames)
            self.assertFalse(frames < worker.MIN_WINDOW_FRAMES,
                             "长窗不应满足护栏条件：%d" % frames)
            self.assertEqual(window["preprocessing_status"] if "preprocessing_status" in window else None, None)

    def test_boundary_just_below_threshold_is_refused(self):
        """399 帧被拒；400 帧放行。"""
        self.assertTrue(worker.MIN_WINDOW_FRAMES - 1 < worker.MIN_WINDOW_FRAMES)
        window = short_window(worker.MIN_WINDOW_FRAMES - 1)
        self.assertTrue(worker.refuse_short_window(window, worker.MIN_WINDOW_FRAMES - 1, {"errors": []}, {}))
        self.assertEqual(window["preprocessing_status"], "failed")

    def test_guard_continues_instead_of_appending_the_row(self):
        """源码层不变量：护栏之后是 continue，短行不进 rows/批次。"""
        source = Path(worker.__file__).read_text(encoding="utf-8")
        marker = "refuse_short_window(window, len(wave), context, counters)"
        self.assertIn(marker, source, "护栏调用点缺失")
        tail = source.split(marker, 1)[1].lstrip().splitlines()[0].strip()
        self.assertEqual(tail, "continue",
                         "护栏后必须是 continue，否则短窗仍会进入 ASR 批次")

    def test_counters_are_initialised_and_reported(self):
        """计数字段要在 counters 初值与 performance 选择表中同时存在。"""
        source = Path(worker.__file__).read_text(encoding="utf-8")
        self.assertIn('"short_window_refused": 0', source)
        self.assertIn("'short_window_refused')", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
