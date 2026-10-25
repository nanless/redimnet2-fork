"""Small CPU-only identity and atomic-control primitives for GPU execution."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import signal
import stat
import time


class Refused(RuntimeError):
    pass


def need(ok, message):
    if not ok:
        raise Refused(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, indent=2) + '\n').encode()
    tmp = path.with_name('.' + path.name + '.' + secrets.token_hex(12) + '.tmp')
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # A reader polling the final pathname must never observe partial JSON.
    # Failure retains the complete tmp as evidence and never overwrites a peer.
    os.link(tmp, path, follow_symlinks=False)
    tmp.unlink()
    fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def process_start(pid):
    raw = Path(f'/proc/{pid}/stat').read_text()
    return int(raw[raw.rfind(')') + 2:].split()[19])


def open_pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid, 0)
    need(platform.system() == 'Linux' and platform.machine() == 'x86_64',
         'no verified pidfd API for this architecture')
    header = Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
    need(re.search(r'^#define\s+__NR_pidfd_open\s+434\s*$', header, re.M),
         'pidfd syscall number not verified')
    lib = ctypes.CDLL(None, use_errno=True)
    lib.syscall.restype = ctypes.c_long
    fd = lib.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open failed')
    os.set_inheritable(fd, False)
    info = Path(f'/proc/self/fdinfo/{fd}').read_text()
    try:
        need(re.search(r'^Pid:\s*' + str(pid) + r'\s*$', info, re.M), 'pidfd identity differs')
        need(os.readlink(f'/proc/self/fd/{fd}') == 'anon_inode:[pidfd]', 'not a real pidfd')
    except BaseException:
        os.close(fd)
        raise
    return int(fd)


def verify_bundle(root, expected):
    root = Path(root)
    path = root / 'bundle_files.json'
    need(sha(path) == expected, 'external bundle pin differs')
    files = json.loads(path.read_text())
    for name, digest in files.items():
        rel = Path(name)
        need(not rel.is_absolute() and '..' not in rel.parts, 'unsafe bundle member')
        p = root / rel
        need(p.is_file() and not p.is_symlink() and sha(p) == digest, 'bundle member differs: ' + name)
    return files


def model_facts(contract, full_hash=True):
    facts = {}
    for spec in contract['models'].values():
        base = Path(spec['path'])
        need(base.is_absolute(), 'model path not absolute')
        for name, value in spec['files'].items():
            p = base / name
            st = p.stat()
            need(st.st_size == value['bytes'], 'model bytes differ: ' + str(p))
            if full_hash:
                need(sha(p) == value['sha256'], 'model SHA differs: ' + str(p))
            facts[str(p)] = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    for name, digest in contract.get('source_files', {}).items():
        need(sha(name) == digest, 'external inference source differs: ' + name)
    return facts


def verify_model_stats(facts):
    for name, expected in facts.items():
        st = Path(name).stat()
        need([st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns] == expected,
             'model file changed: ' + name)


def own_members(child, started, nonce):
    result = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc / 'stat').read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            if fields[0] == 'Z' or int(fields[2]) != child.pid:
                continue
            need(int(fields[3]) == child.pid and int(fields[19]) >= started,
                 'unproven process in own session')
            need(('PILOT_NONCE=' + nonce).encode() in (proc / 'environ').read_bytes().split(b'\0'),
                 'unproven nonce in own session')
            result.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def cleanup(child, started, nonce):
    """Never release a device lease while this owned session might remain."""
    def group_exists():
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                raw = (proc / 'stat').read_text()
                f = raw[raw.rfind(')') + 2:].split()
                if f[0] != 'Z' and int(f[2]) == child.pid:
                    return True
            except (FileNotFoundError, ProcessLookupError):
                pass
        return False
    try:
        if own_members(child, started, nonce):
            os.killpg(child.pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while group_exists() and time.monotonic() < deadline:
            child.poll()
            time.sleep(.1)
        if own_members(child, started, nonce):
            os.killpg(child.pid, signal.SIGKILL)
    except Exception as exc:
        print('OWN_CLEANUP_UNPROVEN_KEEPING_LOCKS ' + repr(exc), flush=True)
    while group_exists():
        child.poll()
        time.sleep(2)


class DrainState:
    """The attested row may only disappear after ACK; it cannot reappear."""
    def __init__(self, uuids, opaque_pid):
        self.uuids = list(uuids)
        self.opaque_pid = opaque_pid
        self.disappeared = set()

    def observe(self, snapshot):
        for uuid in self.uuids:
            rows = [p for p in snapshot['compute_processes'] if p['gpu_uuid'] == uuid]
            need(len(rows) <= 1, 'additional compute row during terminal drain')
            if not rows:
                self.disappeared.add(uuid)
            for row in rows:
                need(uuid not in self.disappeared, 'compute row reappeared after drain')
                need(row['pid'] == self.opaque_pid and type(row['used_mib']) is int
                     and row['used_mib'] >= 0, 'foreign/unknown compute row during drain')


def wait_attested_context_drain(states, capture, timeout=30., interval=.5,
                                clock=time.monotonic, sleep=time.sleep, on_snapshot=None):
    """After owned processes exit, wait only for already-attested rows to vanish.

    Does not admit resources or send signals. Unknown/additional/reappearing
    rows fail. The caller must still run a separate >=5-second idle double probe.
    """
    import math
    uuids = [u for state in states for u in state.uuids]
    need(states and all(state.uuids for state in states) and len(set(uuids)) == len(uuids), 'ambiguous terminal drain bindings')
    need(0 < timeout <= 60 and 0 < interval <= 2, 'invalid terminal drain bound')
    started = clock(); snapshots = []; last_stamp = None
    while True:
        before = clock(); snapshot = capture(); after = clock()
        stamp = snapshot.get('monotonic')
        need(type(stamp) in (int, float) and math.isfinite(stamp)
             and before - .02 <= stamp <= after + .02
             and (last_stamp is None or stamp > last_stamp), 'terminal drain snapshot stale/future')
        last_stamp = stamp; snapshots.append(snapshot)
        if on_snapshot: on_snapshot(snapshot)
        for state in states: state.observe(snapshot)
        need(clock() - started <= timeout, 'terminal drain bound exceeded during capture')
        if all(set(state.uuids) == state.disappeared for state in states):
            return {'all_attested_contexts_absent': True, 'elapsed_seconds': clock() - started,
                    'snapshots': snapshots, 'PID_translation_claimed': False,
                    'idle_admission_proved': False}
        need(clock() - started < timeout, 'attested CUDA context did not disappear within terminal drain bound')
        sleep(interval)
