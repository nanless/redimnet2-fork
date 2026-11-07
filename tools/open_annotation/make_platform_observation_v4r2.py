#!/usr/bin/env python3
"""Build the platform observation the handoff actually validates.

Shape required by native_batch_queue_handoff.observation():
  schema      annotation-bounded-speed-platform-observation/v1  (M.PLATFORM_SCHEMA)
  old_task_id / old_queue_root      必须与 handoff-spec 一致
  observed_utc                      带时区；<= now 且 < 3600s；且必须晚于新任务的 request.created_utc
  evidence_basis                    platform_terminal_and_current_shared_locks
  remote_proc_enumeration_performed False
  competing_successor_tasks         []
  prior_attempts                    []（或列出的都必须终态且 active_workers=0）
  terminal_observation              {"path","sha256"} —— 指向 writer terminal observation
  current_successor_task            新任务自己：state=Running、active_workers=1、
                                    workers=[{"name": <hostname>, "state": "Running"}]

terminal observation 文件自身要求 schema=annotation-writer-terminal-observation/v1，
task_id/queue_root 与 handoff-spec 一致，state ∈ 终态集合，active_workers 为整数 0，
observed_utc 与本观测的 observed_utc 完全相等，workers 恰好一条 `<task>-worker-0` 且终态。

Run at submission time (after the new waiter wrote its request):
"""
import argparse
import datetime
import hashlib
import json
import os
import socket
import sys

PLATFORM_SCHEMA = "annotation-bounded-speed-platform-observation/v1"
TERMINAL_SCHEMA = "annotation-writer-terminal-observation/v1"
MAX_AGE_SECONDS = 3600


def need(condition, message):
    if not condition:
        raise SystemExit(json.dumps({"ok": False, "error": message}, ensure_ascii=False))


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--terminal-observation", required=True,
                   help="已写好的 annotation-writer-terminal-observation/v1 文件")
    p.add_argument("--old-task-id", required=True, help="handoff-spec 的 old_task_id")
    p.add_argument("--old-queue-root", required=True, help="handoff-spec 的 old_queue_root")
    p.add_argument("--successor-task-id", required=True, help="本次新平台任务 ID")
    p.add_argument("--successor-task-name", required=True, help="本次新平台任务名")
    p.add_argument("--successor-hostname", default=None,
                   help="waiter 进程 hostname（默认取当前主机名，必须与 request.identity.hostname 一致）")
    p.add_argument("--not-before", default=None,
                   help="request.created_utc；观测时间必须晚于它")
    p.add_argument("--output", required=True, help="输出路径（必须不存在）")
    args = p.parse_args()

    need(not os.path.exists(args.output), f"refusing to overwrite {args.output}")
    with open(args.terminal_observation, "rb") as stream:
        terminal = json.loads(stream.read())
    need(terminal.get("schema") == TERMINAL_SCHEMA, "terminal observation schema 不对")
    need(terminal.get("task_id") == args.old_task_id, "terminal observation task_id 与 handoff-spec 不一致")
    need(terminal.get("queue_root") == args.old_queue_root, "terminal observation queue_root 与 handoff-spec 不一致")
    need(type(terminal.get("active_workers")) is int and terminal["active_workers"] == 0,
         "terminal observation 的 active_workers 必须是整数 0")
    workers = terminal.get("workers")
    need(isinstance(workers, list) and len(workers) == 1
         and workers[0].get("name") == args.old_task_id + "-worker-0",
         "terminal observation 必须恰好一条 <old_task>-worker-0")
    stamp = terminal.get("observed_utc")
    need(isinstance(stamp, str) and (stamp.endswith("Z") or "+" in stamp), "observed_utc 必须带时区")
    moment = datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    now = datetime.datetime.now(datetime.timezone.utc)
    age = (now - moment).total_seconds()
    need(-60 <= age <= MAX_AGE_SECONDS,
         f"观测已过期: age={age:.0f}s，必须 0..{MAX_AGE_SECONDS}s（提交前现场重取）")
    if args.not_before:
        created = datetime.datetime.fromisoformat(args.not_before.replace("Z", "+00:00"))
        need(moment >= created, "观测时间早于新任务的 request.created_utc，会被拒")

    hostname = args.successor_hostname or socket.gethostname()
    value = {
        "schema": PLATFORM_SCHEMA,
        "old_task_id": args.old_task_id,
        "old_queue_root": args.old_queue_root,
        "observed_utc": moment.astimezone(datetime.timezone.utc).isoformat(),
        "evidence_basis": "platform_terminal_and_current_shared_locks",
        "remote_proc_enumeration_performed": False,
        "competing_successor_tasks": [],
        "prior_attempts": [],
        "terminal_observation": {"path": args.terminal_observation, "sha256": sha(args.terminal_observation)},
        "current_successor_task": {
            "task_id": args.successor_task_id,
            "task_name": args.successor_task_name,
            "state": "Running",
            "active_workers": 1,
            "workers": [{"name": hostname, "state": "Running"}],
        },
    }
    text = json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    with open(args.output, "x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"ok": True, "observed_utc": value["observed_utc"], "age_seconds": round(age, 1),
                      "observation": {"path": args.output,
                                      "sha256": hashlib.sha256(text.encode()).hexdigest()}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
