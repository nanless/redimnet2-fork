#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把仓库里会漂移的状态生成到 docs/STATUS.md。

为什么要有这个脚本
------------------
文档里一旦写死 HEAD、权限位、测试计数、产物 sha、"已修/已验证"、目录里有什么，
它就必然在下次提交后变成假话。本仓库已经因为这件事被审计出 31 条假断言。

做法：易变事实只允许出现在 docs/STATUS.md，而这个文件由本脚本生成。
正文只写"用这条命令查"或"见 STATUS.md 第 N 节"。

三种用法
--------
  python3 tools/gen_status.py            # 生成/刷新 docs/STATUS.md（含 pytest）
  python3 tools/gen_status.py --fast     # 同上，但跳过 pytest（把测试节写"未跑：--fast"）
  python3 tools/gen_status.py --check    # 只校验 STATUS.md 是否已过期；过期则退出码 1
  python3 tools/gen_status.py --lint     # 扫描文档里的易变硬编码断言；有命中则退出码 1

约束
----
* 纯标准库；不联网；只写仓库内 docs/STATUS.md；只读仓库外实验目录（从不写）。
* 输出稳定：不写生成时间戳，唯一的"时间"来自 git 与实验产品的文件时间，因此可 diff。
* --check 会复用 STATUS.md 里已存的测试节（不重跑 pytest），其余各节重新计算后逐行比对。

约定
----
* 被测 pytest 套件用 /root/miniforge3/envs/USE/bin/python；找不到就照实写"未跑：<原因>"。
* 文档正文如果确实必须写某个易变数字，用 <!-- volatile-ok: 原因 --> 就地豁免 lint。
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATUS_PATH = REPO / "docs" / "STATUS.md"

# 仓库外的只读证据根（本机不存在时各节写"无本地留档"，不报错）
EXP_ROOT = Path("/root/experiments/redimnet2")
B6_ROOT = EXP_ROOT / "b6-mix-official-20261019"
ANNOT_ROOT = EXP_ROOT / "open-annotation-ultra-20261025"
V4R2 = ANNOT_ROOT / "native-speed-successor-20261028-v4-r2"

PYTEST_PY = "/root/miniforge3/envs/USE/bin/python"
PYTEST_DIR = "tools/open_annotation/tests"
PYTEST_TIMEOUT = 1800

TITLE = "# 仓库状态（自动生成，不手改）"
PREAMBLE = """\
> 本文件由 `tools/gen_status.py` 生成。正文不复制这里的数字：正文写“见
> [STATUS.md](STATUS.md) 第 N 节”或复核命令，由读者自己查。
> 重新生成：`python3 tools/gen_status.py`；校验是否过期：`python3 tools/gen_status.py --check`；
> 检查文档有没有又把易变事实写死：`python3 tools/gen_status.py --lint`。
> 为保证可 diff，本文件不含生成时间戳；“时间”只来自 git 提交与实验产物的文件时间。\
"""

# 五个关键脚本（是否有可执行位属于易变事实，只允许出现在这里）
SCRIPTS = [
    "tools/ptn_resume/runtime/launch.sh",
    "tools/ptn_resume/lm_launch/package_launch.sh",
    "tools/open_annotation/launch_queue_handoff.sh",
    "tools/open_annotation/launch_portable_successor.sh",
    "tools/ptn_reference/archive/launch.sh",
]

LM_DIR = B6_ROOT / "lm-from-ptn115-20261106-v1"
PTN_DIR = B6_ROOT / "ptn-resume-model47-capture-20261026-v1"

# (标签, 路径, 说明)。→ 产物一节按此顺序输出，缺失就写 missing。
ARTIFACTS = [
    ("LM 最终权重", LM_DIR / "lm" / "models" / "model_1.pt"),
    ("PTN 第 115 轮", PTN_DIR / "ptn" / "models" / "model_115.pt"),
    ("PTN 第 120 轮", PTN_DIR / "ptn" / "models" / "model_120.pt"),
    ("v4-r2 冻结包清单", V4R2 / "frozen-native-reviewed-v4-r2" / "bundle_files.json"),
    ("v4-r2 准入副本清单", V4R2 / "admission-20261108" / "formal-code" / "bundle_files.json"),
    ("v4-r2 v4 pilot 副本清单", V4R2 / "pilot-20261108" / "prepared-v6" / "native-fp32-sdpa-b8" / "code" / "bundle_files.json"),
]

# 最近三次标注提交（v2/v3/v4）。目录名固定，任务号取 publish-work 下的文件名。
ANNOT_SUBMISSIONS = [
    ("v2", "handoff-v4r2-20261107-v2", "run-v2"),
    ("v3", "handoff-v4r2-20261108-v3", "run-v3"),
    ("v4", "handoff-v4r2-20261108-v4", "run-v4"),
]

SECTION_ORDER = [
    "1. 仓库",
    "2. 关键脚本可执行位",
    "3. 测试",
    "4. 已知产物",
    "5. 最近三次标注提交",
    "6. 说明",
]
TEST_SECTION = "3. 测试"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def run(cmd, cwd=REPO, timeout=120, env=None):
    """运行一条命令，返回 (rc, stdout, stderr)。命令不存在时 rc=-1。"""
    try:
        p = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, env=env, errors="replace",
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError as exc:
        return -1, "", "command not found: %s" % exc
    except subprocess.TimeoutExpired:
        return -2, "", "timeout after %ss" % timeout


def sha256_of(path: Path, limit_bytes=None):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def human_size(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            if unit == "B":
                return "%d B" % n
            return "%.1f %s" % (n, unit)
        n /= 1024.0


def mtime_iso(path: Path):
    import datetime
    return datetime.datetime.fromtimestamp(
        path.stat().st_mtime, datetime.timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%SZ")


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def section_repo():
    lines = ["## 1. 仓库", ""]
    rc, out, _ = run(["git", "rev-parse", "HEAD"])
    lines.append("- 完整 HEAD：`%s`" % (out.strip() if rc == 0 else "读不到"))
    rc, out, _ = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    lines.append("- 当前分支：`%s`" % (out.strip() if rc == 0 else "读不到"))
    rc, out, _ = run(["git", "log", "-1", "--format=%h %ad %s", "--date=iso"])
    lines.append("- 顶端提交：`%s`" % (out.strip() if rc == 0 else "读不到"))
    rc, out, _ = run(["git", "-c", "core.quotePath=false", "status", "--porcelain"])
    dirty = [ln for ln in out.splitlines() if ln.strip()]
    lines.append("- 未提交改动：%d 项（`git status --porcelain | wc -l`；若本文件刚被改，"
                 "会把自己算进去）" % len(dirty))
    if dirty:
        lines.append("")
        lines.append("```")
        lines.extend(dirty[:20])
        if len(dirty) > 20:
            lines.append("... 其余 %d 项省略" % (len(dirty) - 20))
        lines.append("```")
    lines.append("")
    lines.append("复核：`git rev-parse HEAD`、`git log -1 --format='%h %ad %s'`、"
                 "`git status --porcelain | wc -l`")
    return lines


def section_scripts():
    lines = ["## 2. 关键脚本可执行位", ""]
    lines.append("| 脚本 | 权限 | `test -x` | 能否 `./xxx.sh` |")
    lines.append("|---|---|---|---|")
    for rel in SCRIPTS:
        path = REPO / rel
        if not path.exists():
            lines.append("| `%s` | 不存在 | 否 | 否 |" % rel)
            continue
        mode = oct(path.stat().st_mode & 0o777)[2:]
        executable = os.access(str(path), os.X_OK)
        lines.append("| `%s` | %s | %s | %s |" % (
            rel, mode, "ok" if executable else "否",
            "能" if executable else "不能，用 `bash %s` 调用" % rel))
    lines.append("")
    lines.append("复核：`test -x <file> && echo ok`、`stat -c '%a %n' <file>`、"
                 "`git ls-files -s <file>`（`100755` 即带可执行位）")
    return lines


def pytest_section_text():
    """真跑一次 pytest，返回 (正文行, 通过/失败摘要字符串)。"""
    py = Path(PYTEST_PY)
    if not py.exists():
        return (["未跑：解释器 `%s` 不存在。" % PYTEST_PY], "未跑：解释器缺失")
    if not (REPO / PYTEST_DIR).is_dir():
        return (["未跑：目录 `%s` 不存在。" % PYTEST_DIR], "未跑：目录缺失")
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    rc, out, err = run(
        [str(py), "-m", "pytest", PYTEST_DIR, "-q", "-p", "no:cacheprovider"],
        timeout=PYTEST_TIMEOUT, env=env,
    )
    tail = (out.strip().splitlines() or [""])[-1].strip()
    summary = re.sub(r"\s+in\s+[\d.]+s.*$", "", tail).strip()
    body = []
    if rc == -2:
        body.append("未跑：pytest 超过 %s 秒未结束。" % PYTEST_TIMEOUT)
        return body, "未跑：超时"
    if rc == -1:
        body.append("未跑：`%s -m pytest` 起不来。" % PYTEST_PY)
        return body, "未跑：无法启动"
    body.append("命令：`%s -m pytest %s -q -p no:cacheprovider`" % (PYTEST_PY, PYTEST_DIR))
    body.append("")
    body.append("- 退出码：%d" % rc)
    body.append("- 摘要行：`%s`" % (summary or "（没有输出摘要行）"))
    body.append("")
    body.append("注意：这是本次生成时的观测。用例数、失败项都会变，引用前先重跑。")
    if rc != 0:
        fails = [ln.strip() for ln in out.splitlines() if ln.startswith("FAILED ")]
        if fails:
            body.append("")
            body.append("失败用例：")
            for ln in fails[:20]:
                body.append("- `%s`" % ln[len("FAILED "):])
        if not summary:
            body.append("")
            body.append("stderr 末尾：")
            body.append("")
            body.append("```")
            body.extend(err.strip().splitlines()[-10:])
            body.append("```")
    return body, (summary or "无摘要")


def section_tests(use_cached_text=None):
    lines = ["## 3. 测试", ""]
    if use_cached_text is not None:
        body = list(use_cached_text)
    else:
        body, _ = pytest_section_text()
    # 去掉首尾空行：否则 build_status 自己补的空行会叠加，出现"两个空行"的抖动，
    # 让 --check 每次都判过期（diff 只差空行）。
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    lines.extend(body)
    return lines


def artifact_row(label, path: Path):
    if not path.exists():
        return "| %s | `%s` | missing | — | — |" % (label, short(path))
    if path.is_dir():
        digest = sha256_of(path / "bundle_files.json") if (path / "bundle_files.json").exists() else ""
        return "| %s | `%s` | 目录 | %s | %s |" % (
            label, short(path),
            "bundle_files.json sha256 `%s`" % digest[:16] if digest else "无清单",
            mtime_iso(path))
    digest = sha256_of(path)
    return "| %s | `%s` | %s | `%s` | %s |" % (
        label, short(path), human_size(path.stat().st_size), digest[:16], mtime_iso(path))


def short(path: Path):
    s = str(path)
    for prefix, repl in (("/root/experiments/redimnet2/", "$EXP/"),
                         ("/root/experiments/", "$EXPROOT/")):
        if s.startswith(prefix):
            return repl + s[len(prefix):]
    return s


def section_artifacts():
    lines = ["## 4. 已知产物", ""]
    lines.append("sha256 只给前 16 位，够 diff；要完整值用下面第 6 节的命令。")
    lines.append("")
    lines.append("| 产物 | 路径 | 大小 | 摘要 | 文件时间 |")
    lines.append("|---|---|---|---|---|")
    for label, path in ARTIFACTS:
        lines.append(artifact_row(label, path))

    # LM 受控包 / 冻结包的记录值（从回执里读，不猜）
    receipt = LM_DIR / "package-r3" / "build-receipt.json"
    if receipt.exists():
        try:
            d = json.loads(receipt.read_text(encoding="utf-8"))
            lines.append("| LM 受控包 r3 | `%s` | %s members | package `%s` / contract `%s` | %s |" % (
                short(receipt.parent), d.get("members", "?"),
                str(d.get("package_sha256", ""))[:16], str(d.get("contract_sha256", ""))[:16],
                mtime_iso(receipt)))
        except Exception as exc:  # noqa: BLE001 - 读不到就照实写
            lines.append("| LM 受控包 r3 | `%s` | 回执解析失败 | %s | — |" % (short(receipt), exc))
    else:
        lines.append("| LM 受控包 r3 | `%s` | missing | — | — |" % short(receipt))

    # v4-r3：本机没有这个目录时也要显式写出来，避免"文档以为有"
    v4r3 = sorted(EXP_ROOT.glob("open-annotation-ultra-*/native-speed-successor-*v4-r3"))
    if v4r3:
        for d in v4r3:
            lines.append("| v4-r3 冻结包 | `%s` | 目录 | 见其中 bundle_files.json | %s |" % (
                short(d), mtime_iso(d)))
    else:
        lines.append("| v4-r3 冻结包 | 无此目录 | missing | — | — |")

    lines.append("")
    lines.append("复核：`sha256sum <artifact>`、`stat -c '%s %y' <artifact>`；"
                 "包身份用其中的 `bundle_files.json`（pilot 与正式包是否同代码，"
                 "用 `diff -rq` 比两个 code 目录，不要比目录名）")
    return lines


def failure_one_liner(run_dir: Path):
    """从 run 目录里的 failure.json / receipt 读一句话失败原因。"""
    if not run_dir.exists():
        return "无本地留档（目录不存在）"
    fails = sorted(run_dir.rglob("failure.json"))
    for f in fails:
        try:
            tb = json.loads(f.read_text(encoding="utf-8")).get("traceback", "")
        except Exception:  # noqa: BLE001
            continue
        tail = [ln.strip() for ln in tb.strip().splitlines() if ln.strip()]
        if tail:
            return "`%s`（%s）" % (tail[-1], f.relative_to(run_dir))
    receipts = sorted(run_dir.rglob("*receipt*.json"))
    for f in receipts:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        code = d.get("training_exit_code", d.get("exit_code"))
        if code is not None:
            return "回执 `%s` 记退出码 %s" % (f.relative_to(run_dir), code)
    if fails or receipts:
        return "有失败原件但读不出一句话原因"
    if not any(run_dir.iterdir()):
        return "无本地留档（run 目录为空）"
    return "无本地留档（没有 failure.json / 回执）"


def section_submissions():
    lines = ["## 5. 最近三次标注提交", ""]
    lines.append("任务号从目录名与 `publish-work/*.json` 的文件名取；"
                 "平台侧终态本地**没有**留档，`terminal-*.json` 里的 `task_id` 字段已知会"
                 "错绑旧任务，不能当新任务的终态用。")
    lines.append("")
    lines.append("| 代次 | 交接目录 | 任务号（取文件名） | 失败原因一句话 | 侧车条数 |")
    lines.append("|---|---|---|---|---|")
    for gen, dirname, rundir in ANNOT_SUBMISSIONS:
        base = ANNOT_ROOT / dirname
        if not base.exists():
            lines.append("| %s | `%s` | 无本地留档 | 目录不存在 | — |" % (gen, dirname))
            continue
        ids = sorted(p.stem.replace("terminal-", "") for p in (base / "publish-work").glob("terminal-*.json")) \
            if (base / "publish-work").is_dir() else []
        task = ids[0] if ids else "无本地留档"
        run_dir = base / rundir
        reason = failure_one_liner(run_dir)
        sidecars = len(list(run_dir.rglob("labels/*.json"))) if run_dir.exists() else 0
        lines.append("| %s | `%s` | `%s` | %s | %d |" % (gen, dirname, task, reason, sidecars))
    lines.append("")
    lines.append("复核：`ls /root/experiments/redimnet2/open-annotation-ultra-*/handoff-*`；"
                 "`find <run> -path '*labels*' -type f | wc -l`；`cat <run>/resources-attempt1/failure.json`")
    return lines


def section_notes():
    return [
        "## 6. 说明",
        "",
        "- 本文件只用于承载易变事实。`docs/*.md` 正文不写死 HEAD、权限位、"
        "测试计数、产物 sha、“已修/已验证”、目录里有什么。",
        "- 复核全部产物 sha：",
        "",
        "  ```bash",
        "  sha256sum /root/experiments/redimnet2/b6-mix-official-20261019/lm-from-ptn115-20261106-v1/lm/models/model_1.pt \\",
        "            /root/experiments/redimnet2/b6-mix-official-20261019/ptn-resume-model47-capture-20261026-v1/ptn/models/model_{115,120}.pt",
        "  ```",
        "",
        "- 本文件由 `python3 tools/gen_status.py` 生成；`--check` 判过期，`--lint` 查正文硬编码。"
        "两者都不需要 GPU、不联网、不写仓库外路径。",
        "",
    ]


def build_status(use_cached_test_text=None):
    lines = [TITLE, "", PREAMBLE, "", "---", ""]
    sections = {
        "1. 仓库": section_repo(),
        "2. 关键脚本可执行位": section_scripts(),
        "3. 测试": section_tests(use_cached_test_text),
        "4. 已知产物": section_artifacts(),
        "5. 最近三次标注提交": section_submissions(),
        "6. 说明": section_notes(),
    }
    for name in SECTION_ORDER:
        lines.extend(sections[name])
        lines.append("")
    text = "\n".join(lines).rstrip("\n") + "\n"
    return text


# --------------------------------------------------------------------------- #
# --check / --lint
# --------------------------------------------------------------------------- #
def split_sections(text):
    """按 '## N. 标题' 切块，返回 {标题: [行]}。"""
    out, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^## (\d+\. .+?)\s*$", line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        if cur is not None:
            out[cur].append(line)
    return out


def cmd_check():
    if not STATUS_PATH.exists():
        print("STATUS.md 不存在：%s" % STATUS_PATH)
        return 1
    current = STATUS_PATH.read_text(encoding="utf-8")
    cached_tests = split_sections(current).get(TEST_SECTION)
    fresh = build_status(use_cached_test_text=cached_tests)
    if fresh == current:
        print("STATUS.md 与现场一致（测试节按已存内容比较，未重跑 pytest）。")
        return 0
    diff = list(difflib.unified_diff(
        current.splitlines(), fresh.splitlines(),
        fromfile="docs/STATUS.md（当前）", tofile="docs/STATUS.md（现场重算）", lineterm=""))
    print("STATUS.md 已过期，需重新生成：python3 tools/gen_status.py")
    print("差异（前 80 行）：")
    for line in diff[:80]:
        print(line)
    if len(diff) > 80:
        print("... 其余 %d 行省略" % (len(diff) - 80))
    return 1


# (名字, 正则, 建议, 是否允许"同行带日期"豁免)
# 允许日期豁免的三类：测试计数、任务状态、"已修"——它们写成"截至 <日期> 的观测"是合规的。
# HEAD、工作树、权限位不给日期豁免：这几项必须换成命令，日期也救不了。
LINT_RULES = [
    ("head_sha", re.compile(r"HEAD\s*(?:为|是|=|：|:)?\s*`?[0-9a-f]{7,40}`?"),
     "正文不要写死 HEAD；改成 `git rev-parse --short HEAD` 或见 STATUS.md 第 1 节", False),
    ("clean_tree", re.compile(r"工作[树区]\s*干净"),
     "工作树是否干净是易变事实；改成 `git status --porcelain | wc -l`", False),
    ("exec_bit", re.compile(r"没[有]?可执行位|是\s*644|（\*\*644\*\*）|`644`|；644"),
     "权限位是易变事实；改成 `test -x <file> && echo ok` 或见 STATUS.md 第 2 节", False),
    ("exit_code", re.compile(r"退出码\s*=?\s*255|均报.*退出码"),
     "退出码随 CLI 版本变；改成\"打印 <原话>，shell 退出码以实测为准\"并给命令", False),
    ("test_count", re.compile(r"\d{2,4}\s*项[^。，]{0,12}(?:通过|回归)|\b\d+\s+passed\b"),
     "测试计数是易变事实；改成 `pytest -q | tail -1` 或见 STATUS.md 第 3 节", True),
    ("fixed_claim", re.compile(r"已修好|已经修好|已彻底修复"),
     "\"已修\"会过期；改成\"截至 <日期> 的观测\"并给复核命令", True),
    ("queue_state", re.compile(r"排队中|仍在排队"),
     "任务状态会过期；改成任务目录/回执检查命令，并写清观测时间", True),
    ("dir_listing", re.compile(r"共\s*\d+\s*篇|目录里(?:有|共)\s*\d+"),
     "目录清点是易变事实；改成 `ls docs/*.md | wc -l` 或见 STATUS.md", False),
]
LINE_REF = re.compile(r"\b(\d{2})[:：](\d{2,4})\b")
EXEMPT = re.compile(r"<!--\s*volatile-ok[:：]\s*(.+?)\s*-->")
# 同行出现以下任一日期写法，即视为"截至某日的观测"，允许 test_count/queue_state/fixed_claim
DATED = re.compile(r"20\d{2}-\d{2}-\d{2}|\d{1,2}\s*月\s*\d{1,2}\s*日|\b\d{2}-\d{2}\b")
# 文件名带日期的参考件是历史快照（如 xxx-20261025.md），允许保留当时的数字
DATED_FILE = re.compile(r"-\d{8}\.md$")
# lint 默认只扫"入口/正文"文档：docs 顶层 + 各技能 SKILL.md。
# 技能 references/ 是带日期的记录与长文存档，不在默认管辖内（要扫就加 --lint-all）。
SCAN_GLOBS = ["docs/*.md", ".agents/skills/*/SKILL.md"]
SCAN_GLOBS_ALL = ["docs/*.md", ".agents/skills/**/*.md"]
# 文件级豁免：某篇整篇都在"引用旧说法"时用，声明后 lint 会把它列出来让人审
FILE_ALLOW = re.compile(r"<!--\s*lint-allow[:：]\s*([^>]*?)\s*-->")


def cmd_lint(scan_all=False):
    violations = []
    exemptions = []
    for pattern in (SCAN_GLOBS_ALL if scan_all else SCAN_GLOBS):
        for path in sorted(REPO.glob(pattern)):
            if path.resolve() == STATUS_PATH.resolve():
                continue  # STATUS.md 就是给易变事实用的
            if DATED_FILE.search(path.name):
                continue  # 日期命名的参考件=历史快照，允许写当时的数字
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            file_allowed = set()
            for line in lines:
                m = FILE_ALLOW.search(line)
                if m:
                    file_allowed.update(x.strip() for x in m.group(1).split(",") if x.strip())
            if file_allowed:
                exemptions.append((str(path.relative_to(REPO)), sorted(file_allowed)))
            for idx, line in enumerate(lines, 1):
                if EXEMPT.search(line):
                    continue
                dated = bool(DATED.search(line))
                for name, rx, advice, date_ok in LINT_RULES:
                    if dated and date_ok:
                        continue
                    if name in file_allowed:
                        continue
                    m = rx.search(line)
                    if not m:
                        continue
                    # 写成 `排队中` 这种反引号片段时是在引用平台状态标签，不是断言
                    if name in ("queue_state", "fixed_claim"):
                        i = m.start()
                        if i > 0 and line[i - 1] == "`" and line[m.end():m.end() + 1] == "`":
                            continue
                    violations.append((str(path.relative_to(REPO)), idx, name, m.group(0), advice))
                for m in LINE_REF.finditer(line):
                    if int(m.group(2)) >= 60:  # 排除 13:14 这类时刻
                        violations.append((str(path.relative_to(REPO)), idx, "line_ref",
                                           m.group(0),
                                           "别用 文件:行号 指向别处（一改就漂）；改用小节标题或命令"))
    if exemptions:
        print("生效中的文件级豁免（每次都在这里列出来，便于复核）：")
        for f, rules in exemptions:
            print("  %s → %s" % (f, ", ".join(rules)))
        print("")
    if not violations:
        print("未发现易变硬编码断言（%s）。" % "、".join(SCAN_GLOBS_ALL if scan_all else SCAN_GLOBS))
        return 0
    print("发现 %d 处易变硬编码断言：" % len(violations))
    for f, ln, name, snippet, advice in violations:
        print("  %s:%d  [%s] %r" % (f, ln, name, snippet))
        print("      → %s" % advice)
    print("")
    print("确实必须写时，在同行加 <!-- volatile-ok: 原因 --> 豁免。")
    return 1


# --------------------------------------------------------------------------- #
def cmd_generate(fast=False):
    if fast:
        test_text = ["未跑：本次用 `--fast` 生成，只刷新其余各节。",
                     "要拿真实摘要请 `python3 tools/gen_status.py`（会跑 %s）。" % TEST_SECTION]
        text = build_status(use_cached_test_text=test_text)
    else:
        text = build_status()
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    old = STATUS_PATH.read_text(encoding="utf-8") if STATUS_PATH.exists() else None
    STATUS_PATH.write_text(text, encoding="utf-8")
    if old is None:
        print("已生成 %s（%d 行）" % (STATUS_PATH.relative_to(REPO), len(text.splitlines())))
    elif old == text:
        print("%s 无变化（%d 行）" % (STATUS_PATH.relative_to(REPO), len(text.splitlines())))
    else:
        print("已刷新 %s（%d 行，内容有变化）" % (STATUS_PATH.relative_to(REPO), len(text.splitlines())))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="把易变状态生成到 docs/STATUS.md，并提供过期校验与文档 lint。")
    ap.add_argument("--check", action="store_true", help="校验 STATUS.md 是否过期（退出码 1=过期）")
    ap.add_argument("--lint", action="store_true", help="扫描 docs 与技能 SKILL.md 里写死的易变事实（退出码 1=有命中）")
    ap.add_argument("--lint-all", action="store_true", help="连技能的 references/ 一起扫（默认不扫）")
    ap.add_argument("--fast", action="store_true", help="生成时跳过 pytest")
    args = ap.parse_args(argv)
    if args.check and (args.lint or args.lint_all):
        ap.error("--check 与 --lint 一次只做一个")
    if args.check:
        return cmd_check()
    if args.lint or args.lint_all:
        return cmd_lint(scan_all=args.lint_all)
    return cmd_generate(fast=args.fast)


if __name__ == "__main__":
    sys.exit(main())
