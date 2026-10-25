"""Synthetic edge cases; these scores are not model quality measurements."""
import importlib.util
import csv
import json
import math
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np

from duration_ft.metrics import (score_metrics, fixed_threshold_metrics,
                                 calibrate_far_threshold, paired_speaker_bootstrap,
                                 single_speaker_bootstrap)


class MetricTests(unittest.TestCase):
    def test_perfect_and_inverted(self):
        a = score_metrics([0, 0, 1, 1], [.1, .2, .8, .9])
        self.assertEqual((a["eer"], a["auc"]), (0, 1))
        self.assertEqual(a["min_dcf"]["0.001"]["value"], 0)
        b = score_metrics([1, 1, 0, 0], [.1, .2, .8, .9])
        self.assertEqual((b["eer"], b["auc"]), (1, 0))
        self.assertEqual(b["min_dcf"]["0.01"]["value"], 1)
        self.assertIsNone(b["min_dcf"]["0.01"]["threshold"])
        self.assertEqual(b["min_dcf"]["0.01"]["threshold_kind"], "reject_all")

    def test_all_tied(self):
        r = score_metrics([0, 1, 0, 1], [.5] * 4)
        self.assertEqual(r["eer"], .5)
        self.assertEqual(r["auc"], .5)
        self.assertTrue(r["eer_interpolated"])
        self.assertIsNone(r["eer_threshold"])

    def test_interpolation_not_a_fractional_tie_decision(self):
        r = score_metrics([1, 0, 0], [.6, .6, .2])
        self.assertAlmostEqual(r["eer"], 1 / 3)
        self.assertAlmostEqual(r["auc"], .75)
        self.assertIsNone(r["eer_threshold"])
        # At the actual tie threshold both the genuine and impostor are accepted.
        f = fixed_threshold_metrics([1, 0, 0], [.6, .6, .2], .6)
        self.assertEqual((f["far"], f["frr"]), (.5, 0))

    def test_min_dcf_normalization(self):
        r = score_metrics([1, 0, 1, 0], [.9, .8, .7, .6])
        self.assertEqual(r["min_dcf"]["0.01"]["value"], .5)
        self.assertEqual(r["min_dcf"]["0.001"]["value"], .5)

    def test_threshold_equal_score_is_accept(self):
        r = fixed_threshold_metrics([1, 0, 1, 0], [.5, .5, .4, .4], .5)
        self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"]), (1, 1, 1, 1))
        self.assertEqual((r["far_percent"], r["frr_percent"]), (50, 50))

    def test_dev_conservative_no_split_tie(self):
        r = calibrate_far_threshold([1, 1, 0, 0], [.95, .85, .9, .9], target_far=.25)
        self.assertEqual(r["threshold"], .95)
        self.assertEqual((r["dev_far"], r["dev_frr"]), (0, .5))

    def test_dev_far_resolution(self):
        y = [0] * 100 + [1, 1]
        s = list(np.arange(100) / 100) + [1.0, .8]
        r = calibrate_far_threshold(y, s, target_far=.01)
        self.assertLessEqual(r["dev_far"], .01)
        self.assertEqual(fixed_threshold_metrics(y, s, r["threshold"])["fp"], 1)

    def test_reject_all_calibration_is_finite_and_json(self):
        r = calibrate_far_threshold([1, 0], [.5, .5], target_far=0)
        self.assertGreater(r["threshold"], .5)
        self.assertTrue(math.isfinite(r["threshold"]))
        json.dumps(r, allow_nan=False)
        self.assertEqual(fixed_threshold_metrics([1, 0], [.5, .5], r["threshold"])["frr"], 1)

    def test_extreme_float_reject_all_unrepresentable(self):
        with self.assertRaisesRegex(ValueError, "finite reject-all"):
            calibrate_far_threshold([1, 0], [np.finfo(float).max] * 2, target_far=0)

    def test_monotone_score_transform_preserves_discrimination(self):
        y, s = [1, 0, 1, 0], [.9, .8, .7, .6]
        a, b = score_metrics(y, s), score_metrics(y, np.array(s) * 3 + .2)
        self.assertEqual(a["eer"], b["eer"])
        self.assertEqual(a["auc"], b["auc"])
        self.assertEqual(a["min_dcf"]["0.01"]["value"], b["min_dcf"]["0.01"]["value"])

    def test_weights_match_explicit_repetition(self):
        y, s, w = [1, 0, 1, 0], [.9, .8, .7, .6], [2, 3, 1, 2]
        a = score_metrics(y, s, weights=w)
        b = score_metrics(np.repeat(y, w), np.repeat(s, w))
        self.assertAlmostEqual(a["eer"], b["eer"])
        self.assertAlmostEqual(a["auc"], b["auc"])

    def test_invalid_input_rejected(self):
        cases = [([], []), ([1], [.1]), ([0, 1], [.1, float("nan")]),
                 ([0, 1], [.1, float("inf")]), ([0, 2], [.1, .2]),
                 ([0, 1], [.1]), ([[0, 1]], [[.1, .2]])]
        for y, s in cases:
            with self.subTest(y=y, s=s), self.assertRaises(ValueError):
                score_metrics(y, s)
        for w in [[1, -1], [0, 1], [float("inf"), 1], [1]]:
            with self.subTest(weights=w), self.assertRaises(ValueError):
                score_metrics([0, 1], [.1, .2], weights=w)

    def test_prior_and_nan_threshold_rejected(self):
        for p in [0, 1, float("nan")]:
            with self.assertRaises(ValueError):
                score_metrics([0, 1], [.1, .2], p_targets=[p])
        with self.assertRaises(ValueError):
            fixed_threshold_metrics([0, 1], [.1, .2], float("nan"))


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.enroll, self.test, self.y, self.s = [], [], [], []
        for i in range(6):
            for k in range(6):
                self.enroll.append(str(i))
                self.test.append(str(k))
                self.y.append(int(i == k))
                self.s.append(.4 + (i % 3) * .1 if i == k else .3 + ((i + k) % 4) * .1)

    def run_boot(self, **kwargs):
        return paired_speaker_bootstrap(self.y, self.s, self.s, self.enroll, self.test,
                                       n_bootstrap=40, seed=12, **kwargs)

    def test_identical_models_zero_paired_difference(self):
        r = self.run_boot(threshold_a=.5, threshold_b=.5)
        self.assertEqual(r["status"], "ok")
        for v in r["differences"].values():
            self.assertEqual((v["estimate"], v["lower"], v["upper"]), (0, 0, 0))
        json.dumps(r, allow_nan=False)

    def test_fixed_seed_reproducible(self):
        self.assertEqual(self.run_boot(), self.run_boot())

    def test_vertex_weights_and_invalid_draws_reported(self):
        r = paired_speaker_bootstrap([1, 1, 0], [.8, .7, .4], [.8, .7, .4],
                                    ["a", "b", "a"], ["a", "b", "b"], n_bootstrap=40, seed=3)
        self.assertGreater(r["invalid_missing_class"], 0)
        self.assertEqual(r["attempts"], r["valid"] + r["invalid_missing_class"])
        self.assertEqual(r["target_weight"], "m_s")
        self.assertEqual(r["nontarget_weight"], "m_s*m_t")
        self.assertEqual(r["status"], "insufficient_valid_resamples")
        self.assertIsNone(r["differences"]["eer"]["lower"])

    def test_speaker_labels_and_threshold_pair_required(self):
        with self.assertRaisesRegex(ValueError, "identity equality"):
            paired_speaker_bootstrap([1, 0], [.8, .2], [.8, .2], ["a", "a"], ["b", "b"], n_bootstrap=40)
        with self.assertRaisesRegex(ValueError, "both fixed"):
            self.run_boot(threshold_a=.5)


class AbsoluteBootstrapTests(unittest.TestCase):
    """Absolute intervals must not be inferred from the paired difference CI."""

    setUp = BootstrapTests.setUp

    def single(self, scores=None, **kwargs):
        return single_speaker_bootstrap(self.y, self.s if scores is None else scores,
                                        self.enroll, self.test, n_bootstrap=500,
                                        seed=20261023, **kwargs)

    def test_all_tied_absolute_fifty_percent(self):
        r = self.single([.5] * len(self.y))
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["intervals"]["eer_percent"],
                         {"estimate": 50, "lower": 50, "upper": 50, "unit": "percent"})
        self.assertEqual(r["intervals"]["eer"]["lower"], .5)
        self.assertEqual(r["intervals"]["eer"]["upper"], .5)
        json.dumps(r, allow_nan=False)

    def test_perfect_absolute_zero(self):
        r = self.single([.9 if label else .1 for label in self.y])
        for interval in r["intervals"].values():
            self.assertEqual((interval["estimate"], interval["lower"], interval["upper"]), (0, 0, 0))

    def test_single_matches_paired_marginal_exactly(self):
        other = [.5 + (i % 2) * .1 if i == k else .25 + ((i + 2*k) % 5) * .1
                 for i in range(6) for k in range(6)]
        pair = paired_speaker_bootstrap(self.y, self.s, other, self.enroll, self.test,
                                       n_bootstrap=500, seed=20261023,
                                       threshold_a=.5, threshold_b=.55)
        a, b = self.single(threshold=.5), self.single(other, threshold=.55)
        self.assertEqual(a["intervals"], pair["marginal_a"])
        self.assertEqual(b["intervals"], pair["marginal_b"])
        self.assertEqual(a["point"], pair["point_a"])
        self.assertEqual(b["point"], pair["point_b"])
        self.assertEqual(a["valid"], pair["valid"])

    def test_existing_paired_difference_values_unchanged(self):
        # Exact baseline recorded from the pre-extension implementation.
        other = [.5 + (i % 2) * .1 if i == k else .25 + ((i + 2*k) % 5) * .1
                 for i in range(6) for k in range(6)]
        r = paired_speaker_bootstrap(self.y, self.s, other, self.enroll, self.test,
                                     n_bootstrap=500, seed=20261023)
        self.assertEqual(r["differences"], {
            "eer": {"estimate": -0.016666666666666663, "lower": -0.3139772727272727,
                    "upper": 0.3436567164179101, "unit": "fraction_or_normalized_cost"},
            "eer_percent": {"estimate": -1.6666666666666714, "lower": -31.397727272727273,
                            "upper": 34.36567164179101, "unit": "percentage_points"},
            "min_dcf_0.01": {"estimate": 0.0, "lower": -0.8333333333333333,
                             "upper": .5, "unit": "fraction_or_normalized_cost"},
            "min_dcf_0.001": {"estimate": 0.0, "lower": -0.8333333333333334,
                              "upper": .5, "unit": "fraction_or_normalized_cost"}})

    def test_random_scores_seed_replay_and_asymmetric_percentile(self):
        rng = random.Random(71)
        scores = [rng.random() for _ in self.y]
        r = self.single(scores)
        self.assertEqual(r, self.single(scores))
        # Independent replay through the public weighted metrics API.
        draws, invalid, sampler = [], 0, random.Random(20261023)
        y = np.array(self.y, dtype=bool)
        a, b = np.array(self.enroll, dtype=int), np.array(self.test, dtype=int)
        for _ in range(500):
            counts = np.bincount(sampler.choices(range(6), k=6), minlength=6)
            weights = np.where(y, counts[a], counts[a] * counts[b])
            if weights[y].sum() <= 0 or weights[~y].sum() <= 0:
                invalid += 1
                continue
            draws.append(score_metrics(self.y, scores, weights=weights)["eer_percent"])
        expected = np.quantile(draws, [(1-.95)/2, 1-(1-.95)/2]).tolist()
        interval = r["intervals"]["eer_percent"]
        self.assertEqual([interval["lower"], interval["upper"]], expected)
        self.assertEqual(r["invalid_missing_class"], invalid)
        self.assertLessEqual(interval["lower"], interval["estimate"])
        self.assertLessEqual(interval["estimate"], interval["upper"])
        self.assertNotAlmostEqual(interval["estimate"]-interval["lower"],
                                  interval["upper"]-interval["estimate"])

    def test_missing_class_draws_do_not_create_absolute_zero_intervals(self):
        r = single_speaker_bootstrap([1, 1, 0], [.8, .7, .4],
                                     ["a", "b", "a"], ["a", "b", "b"], n_bootstrap=40, seed=3)
        self.assertEqual(r["status"], "insufficient_valid_resamples")
        self.assertEqual(r["valid"]+r["invalid_missing_class"], 40)
        for interval in r["intervals"].values():
            self.assertIsNone(interval["lower"])
            self.assertIsNone(interval["upper"])

    def test_single_bad_identity_and_nan_threshold_refused(self):
        with self.assertRaisesRegex(ValueError, "identity equality"):
            single_speaker_bootstrap([1, 0], [.8, .2], ["a", "a"], ["b", "b"])
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.single(threshold=float("nan"))

    def test_forty_speaker_universe_and_default_contract(self):
        enroll = [str(i) for i in range(40)] * 2
        verify = [str(i) for i in range(40)] + [str((i+1) % 40) for i in range(40)]
        r = single_speaker_bootstrap([1]*40 + [0]*40, [.8]*40 + [.2]*40, enroll, verify)
        self.assertEqual((r["speaker_count"], r["attempts"], r["seed"], r["confidence"]),
                         (40, 500, 20261023, .95))
        self.assertEqual(r["intervals"]["eer_percent"]["upper"], 0)


class PlotGuardTests(unittest.TestCase):
    def test_decimal_prior_prefix_does_not_overwrite_other_prior(self):
        from duration_ft.plots import _save
        class FakeFigure:
            def savefig(self, path, **kwargs):
                Path(path).write_bytes(b"test-only")
        with tempfile.TemporaryDirectory() as d:
            a = _save(FakeFigure(), Path(d) / "minDCF-0.01")
            b = _save(FakeFigure(), Path(d) / "minDCF-0.001")
            self.assertEqual(len(list(Path(d).iterdir())), 4)
            self.assertNotEqual(a["png"], b["png"])
    def test_missing_measurements_do_not_create_zero_plot(self):
        from duration_ft.plots import plot_duration_heatmap
        with tempfile.TemporaryDirectory() as d:
            r = plot_duration_heatmap([{"enroll_duration": "full", "test_duration": "full",
                                       "cohort_id": "test-only", "eer_percent": None}], "eer_percent", Path(d) / "no-data")
            self.assertEqual(r["status"], "unavailable")
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_different_cohorts_and_trialsets_rejected(self):
        from duration_ft.plots import plot_duration_heatmap, plot_relative_full
        rows = [{"enroll_duration": "full", "test_duration": "full", "cohort_id": "A",
                 "trialset_sha256": "abc", "complete": True, "eer_percent": 1.0},
                {"enroll_duration": "1", "test_duration": "1", "cohort_id": "B",
                 "trialset_sha256": "def", "complete": True, "eer_percent": 2.0}]
        with self.assertRaises(ValueError):
            plot_duration_heatmap(rows, "eer_percent", "/not-written")
        rows[1]["cohort_id"] = "A"
        with self.assertRaises(ValueError):
            plot_relative_full(rows, "eer_percent", "/not-written")

    def test_partial_cells_not_compared_to_full(self):
        from duration_ft.plots import plot_relative_full
        with self.assertRaisesRegex(ValueError, "complete"):
            plot_relative_full([{"enroll_duration": "full", "test_duration": "full", "cohort_id": "A",
                                 "trialset_sha256": "abc", "complete": False, "eer_percent": .1}],
                               "eer_percent", "/not-written")


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/summarize_duration_results.py"
        spec = importlib.util.spec_from_file_location("duration_summary_test", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def fixture(self, root, *, partial=False, bad_score=False):
        root.mkdir(parents=True, exist_ok=True)
        score_path = root / "scores.csv"
        fields = ["trial_id", "label", "enroll", "verify", "enroll_speaker", "test_speaker", "status", "score"]
        with score_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, (label, a, b, s) in enumerate([(1, "a", "a", .9), (1, "b", "b", .8), (0, "a", "b", .3), (0, "b", "a", .4)]):
                missing = partial and not label
                writer.writerow({"trial_id": str(i), "label": label, "enroll": "e" + str(i), "verify": "t" + str(i),
                                 "enroll_speaker": a, "test_speaker": b, "status": "missing_embedding" if missing else "ok",
                                 "score": "" if missing else ("nan" if bad_score and i == 0 else s)})
        cell = {"matrix": "all-O", "enroll_duration": "full", "verify_duration": "full", "trialset_sha256": "same",
                "expected_trials": 4, "scores_csv": str(score_path)}
        summary = root / "summary.json"
        summary.write_text(json.dumps({"cells": [cell], "run_identity": {"model": "synthetic-test-only"}}))
        return summary

    def test_saved_scores_recomputed_and_no_dev_calibration_invented(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.fixture(Path(d) / "input")
            r = self.mod.summarize(p, Path(d) / "statistics", make_plots=False)
            self.assertEqual(r["cells"][0]["metrics"]["eer"], 0)
            self.assertEqual(r["cells"][0]["fixed_threshold"]["status"], "unavailable")
            json.dumps(r, allow_nan=False)

    def test_missing_class_after_failures_remains_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.fixture(Path(d) / "input", partial=True)
            r = self.mod.summarize(p, Path(d) / "statistics", make_plots=False)
            self.assertFalse(r["cells"][0]["complete"])
            self.assertIsNone(r["cells"][0]["metrics"])
            self.assertEqual(r["cells"][0]["coverage"], .5)

    def test_successful_nan_is_refused_not_filtered(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.fixture(Path(d) / "input", bad_score=True)
            with self.assertRaisesRegex(ValueError, "finite"):
                self.mod.summarize(p, Path(d) / "statistics", make_plots=False)

    def test_partial_with_both_classes_does_not_report_subset_quality(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.fixture(Path(d) / "input")
            score_path = p.parent / "scores.csv"
            with score_path.open() as f:
                rows = list(csv.DictReader(f))
            rows[-1].update(status="missing_embedding", score="")
            with score_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            r = self.mod.summarize(p, Path(d) / "statistics", make_plots=False)
            self.assertEqual(r["cells"][0]["scored_trials"], 3)
            self.assertIsNone(r["cells"][0]["metrics"])

    def test_stale_score_hash_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.fixture(Path(d) / "input")
            j = json.loads(p.read_text())
            j["cells"][0]["scores_sha256"] = "wrong"
            p.write_text(json.dumps(j))
            with self.assertRaisesRegex(ValueError, "SHA"):
                self.mod.summarize(p, Path(d) / "statistics", make_plots=False)

    def test_same_trial_paths_different_crops_do_not_get_paired_ci(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.fixture(Path(d) / "a")
            q = self.fixture(Path(d) / "b")
            for path, crop in [(p, "crop-a"), (q, "crop-b")]:
                j = json.loads(path.read_text())
                j["run_identity"]["comparison_identity_sha256"] = crop
                path.write_text(json.dumps(j))
            r = self.mod.summarize(p, Path(d) / "statistics", compare_summary=q,
                                   n_bootstrap=20, make_plots=False)
            self.assertEqual(r["cells"][0]["paired_bootstrap"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
