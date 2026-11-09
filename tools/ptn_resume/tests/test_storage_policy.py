#!/usr/bin/env python3
"""Storage-placement guard: big-data defaults stay under ``group-shared``.

User rule (2026-11-09): training and evaluation tools default their big-data
writes to ``/root/group-shared/voiceprint``, never to the two retired roots
listed in ``storage_policy.FORBIDDEN_PREFIXES`` or to another ``/root``
directory.  ``/root`` itself keeps only code repositories and temporary logs.

What this test enforces inside this repository's slice of that rule
(``tools/ptn_resume/**``, ``applications/speaker_rec/scripts/**``,
``applications/speaker_rec/launch.sh`` and the recipe trees):

* no source line mentions one of the retired roots unless it carries the
  whitelist marker ``storage-policy-allow: <reason>`` on the same line or on
  one of the two preceding lines.  Python / shell / YAML write it as
  ``# storage-policy-allow: reason``; Markdown writes it inside an HTML comment
  as ``<!-- storage-policy-allow: reason -->``.  The marked lines are the
  read-only references to artifacts that already exist on the historical disk
  (completed PTN run, frozen package, decoder library) plus coordination paths
  such as the GPU UUID lock directory;
* the shell entry points that create big data still default to the shared
  roots, so a silent revert to ``$REPO_ROOT/exp`` or back to a ``/root`` path
  fails here;
* ``tools/ptn_resume/storage_policy.py`` keeps both roots under
  ``/root/group-shared/voiceprint`` and keeps naming the forbidden prefixes.

The scan is textual and line based on purpose: it is cheap, runs on every
pytest invocation, and does not import or execute the tools it inspects.

Run:
    /root/miniforge3/envs/USE/bin/python -m pytest tools/ptn_resume/tests -q
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tools.ptn_resume import storage_policy as P

REPO = Path(__file__).resolve().parents[3]

SCOPE_DIRS = ("tools/ptn_resume", "applications/speaker_rec/scripts")
SCOPE_FILES = ("applications/speaker_rec/launch.sh",)
SCOPE_TREES = (
    "applications/speaker_rec/recipes",
    "applications/speaker_rec/examples/*/recipes",
)
TEXT_SUFFIXES = (".py", ".sh", ".yaml", ".yml", ".template", ".md")
# storage_policy.py exists to name the forbidden prefixes; it is asserted
# separately instead of being scanned for them.
EXEMPT_FILES = ("tools/ptn_resume/storage_policy.py",)
LOOKBACK = 2

# Spelled in two pieces so this file never matches its own scan.
FORBIDDEN = ("/root/" + "experiments", "/root/" + "codex_artifacts")
MARKER = "storage-policy-allow:"

SHARED_ROOT = "/root/group-shared/voiceprint"
EXPERIMENT_ROOT = SHARED_ROOT + "/experiments/redimnet2"
ARTIFACT_ROOT = SHARED_ROOT + "/artifacts/redimnet2"


def scope_files():
    """Every text file this test owns, as repo-relative paths (sorted)."""
    found = set()
    for relative in SCOPE_DIRS:
        base = REPO / relative
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES and "__pycache__" not in path.parts:
                found.add(path.relative_to(REPO).as_posix())
    for relative in SCOPE_FILES:
        if (REPO / relative).is_file():
            found.add(relative)
    for pattern in SCOPE_TREES:
        for tree in sorted(REPO.glob(pattern)):
            if not tree.is_dir():
                continue
            for path in tree.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    found.add(path.relative_to(REPO).as_posix())
    return sorted(found - set(EXEMPT_FILES))


def read_text(relative):
    return (REPO / relative).read_text(encoding="utf-8", errors="replace")


def unmarked_hits():
    """[(file, lineno, line)] for forbidden roots without an allow marker."""
    offenders = []
    for relative in scope_files():
        lines = read_text(relative).splitlines()
        for index, line in enumerate(lines):
            if not any(prefix in line for prefix in FORBIDDEN):
                continue
            window = lines[max(0, index - LOOKBACK):index + 1]
            if not any(MARKER in neighbour for neighbour in window):
                offenders.append((relative, index + 1, line.strip()))
    return offenders


class ScopeTests(unittest.TestCase):
    def test_scope_actually_resolves_to_a_repository_slice(self):
        files = scope_files()
        self.assertGreater(len(files), 20, "storage-policy scan found almost nothing: %r" % (files,))
        self.assertIn("tools/ptn_resume/lm_launch/run_lm.sh", files)
        self.assertIn("applications/speaker_rec/scripts/cloud/train_b6_mix_cloud_8gpu.sh", files)
        self.assertIn("applications/speaker_rec/scripts/cloud/evaluate_b6_mix_reproduction.sh", files)
        self.assertIn("applications/speaker_rec/launch.sh", files)


class ForbiddenDefaultTests(unittest.TestCase):
    def test_no_unmarked_reference_to_legacy_bigdata_roots(self):
        offenders = unmarked_hits()
        self.assertEqual(
            [],
            offenders,
            "big-data defaults must not point at %s; move the default under %s and "
            "mark a genuine read-only historical reference with `# storage-policy-allow: 原因`"
            % ("/".join(FORBIDDEN), SHARED_ROOT),
        )


class PolicyModuleTests(unittest.TestCase):
    def test_roots_live_under_the_shared_group_disk(self):
        self.assertEqual(EXPERIMENT_ROOT, P.EXPERIMENT_ROOT)
        self.assertEqual(ARTIFACT_ROOT, P.ARTIFACT_ROOT)
        for root in (P.GROUP_ROOT, P.EXPERIMENT_ROOT, P.ARTIFACT_ROOT):
            self.assertTrue(root.startswith(SHARED_ROOT + "/") or root == SHARED_ROOT, root)
            for prefix in FORBIDDEN:
                self.assertFalse(root.startswith(prefix), root)

    def test_forbidden_prefixes_still_name_both_retired_roots(self):
        self.assertEqual(FORBIDDEN, tuple(P.FORBIDDEN_PREFIXES))
        self.assertIn(MARKER, P.ALLOW_MARKER)

    def test_experiment_and_eval_directories_have_the_documented_shape(self):
        self.assertEqual(EXPERIMENT_ROOT + "/b6-mix-ptn", P.experiment_dir("b6-mix-ptn"))
        self.assertEqual(ARTIFACT_ROOT + "/eval-20261109", P.eval_dir("20261109"))
        self.assertEqual(
            ARTIFACT_ROOT + "/eval-20261109/b6-mix-lm",
            P.eval_dir("20261109", "b6-mix-lm"),
        )
        for bad in ("", "a/b", "../escape", ".hidden"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    P.experiment_dir(bad)
        for bad in ("2026-11-09", "2026119", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    P.eval_dir(bad)


class ShellDefaultTests(unittest.TestCase):
    """The entry points that create big data must still default to group-shared."""

    def test_lm_wrapper_defaults_its_output_to_the_shared_experiment_root(self):
        text = read_text("tools/ptn_resume/lm_launch/run_lm.sh")
        self.assertIn('EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-%s}"' % EXPERIMENT_ROOT, text)
        self.assertIn('EXP_DIR="${EXP_DIR:-$EXPERIMENT_ROOT/$EXPERIMENT_NAME/lm}"', text)

    def test_lm_env_template_shows_the_shared_experiment_root(self):
        text = read_text("tools/ptn_resume/lm_launch/env.template")
        shared = P.experiment_dir("lm-from-ptn115") + "/lm"
        self.assertEqual(shared, EXPERIMENT_ROOT + "/lm-from-ptn115/lm")
        self.assertIn("EXP_DIR=%s" % shared, text)

    def test_cloud_train_entry_defaults_to_the_shared_experiment_root(self):
        text = read_text("applications/speaker_rec/scripts/cloud/train_b6_mix_cloud_8gpu.sh")
        self.assertIn('EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-%s}"' % EXPERIMENT_ROOT, text)
        for stage in ("ptn", "lm"):
            self.assertIn('DEFAULT_EXP_DIR="$EXPERIMENT_ROOT/b6-mix-%s"' % stage, text)
        self.assertIn('PTN_EXP_DIR="${PTN_EXP_DIR:-$EXPERIMENT_ROOT/b6-mix-ptn}"', text)
        self.assertNotIn('DEFAULT_EXP_DIR="$REPO_ROOT/exp', text)
        self.assertNotIn('PTN_EXP_DIR="${PTN_EXP_DIR:-$REPO_ROOT/exp', text)

    def test_evaluation_entry_defaults_its_artifacts_to_the_shared_artifact_root(self):
        text = read_text("applications/speaker_rec/scripts/cloud/evaluate_b6_mix_reproduction.sh")
        self.assertIn('ARTIFACT_ROOT="${ARTIFACT_ROOT:-%s}"' % ARTIFACT_ROOT, text)
        self.assertIn('EVAL_DIR="${EVAL_DIR:-$ARTIFACT_ROOT/eval-$(date -u +%Y%m%d)/$(basename "$RUN_ROOT")}"', text)
        self.assertNotIn('EVAL_DIR="$RUN_ROOT/evaluation"', text)


if __name__ == "__main__":
    unittest.main()
