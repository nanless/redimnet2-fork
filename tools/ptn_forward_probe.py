#!/usr/bin/env python3
"""在空闲开发双卡上诊断冻结 PTN 前向；不训练、不更新、不覆盖现场。

保存波形已经做过数据增强。本工具不再次增强，使用同一完整批。每个对照
独立新进程、双快照、加锁并重载 model47，避免 train 模式 BN buffer 和
跨命名空间 PID 污染后续准入。它不是缺失的失败时权重/RNG 的替代品。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            start = time.monotonic()
            h.update(block)
            time.sleep(max(0, len(block) / (16 * 1024**2) - (time.monotonic() - start)))
    return h.hexdigest()


def verify(path, digest):
    p = Path(path)
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('必须提供完整64位 SHA256')
    before = p.stat()
    if p.is_symlink() or not p.is_file() or sha(p) != digest:
        raise ValueError('文件身份不符: ' + str(p))
    after = p.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('读取期间文件变化: ' + str(p))


def import_identity(name, module, source):
    """普通模块绑定文件；合法 namespace 包绑定其唯一冻结目录。"""
    origin = getattr(module, '__file__', None)
    if origin is None:
        paths = list(getattr(module, '__path__', []))
        expected = source / name.replace('.', '/')
        if len(paths) != 1 or Path(paths[0]).resolve() != expected or not expected.is_dir():
            raise ValueError('namespace 包含非冻结目录: ' + name)
        return {'namespace_directory': str(expected), 'file_sha256_applicable': False}
    file = Path(origin).resolve()
    if source not in file.parents:
        raise ValueError('模型子模块从非冻结根导入: ' + name)
    return {'file': str(file), 'sha256': sha(file)}


def main():
    os.umask(0o077)
    sys.dont_write_bytecode = True
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'manifest', 'manifest-sha', 'config', 'config-sha',
                 'checkpoint', 'checkpoint-sha', 'wave', 'wave-sha', 'output', 'lock-root'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--expected-uuids', nargs=2, required=True)
    p.add_argument('--case', choices=('baseline-amp', 'hooked-amp', 'hooked-fp32'), required=True)
    p.add_argument('--no-grad', action='store_true', help='仅作明确标记的显存受限诊断，不能视为原训练前向')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError('必须使用全新输出目录')
    for name in ('manifest', 'config', 'checkpoint', 'wave'):
        verify(getattr(a, name), getattr(a, name + '_sha'))
    source = Path(a.source).resolve()
    entries = json.loads(Path(a.manifest).read_text())
    actual = {str(x.relative_to(source)) for x in source.rglob('*') if x.is_file()}
    if actual != {x['relative'] for x in entries}:
        raise ValueError('冻结源码成员不符')
    for entry in entries:
        rel = Path(entry['relative'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('源码清单路径不安全')
        verify(source / rel, entry['sha256'])
        file_stat = (source / rel).stat()
        if file_stat.st_size != entry['bytes'] or stat.S_IMODE(file_stat.st_mode) != entry['mode']:
            raise ValueError('冻结源码长度或权限不符: ' + str(rel))
    # 此导入仅使用审核中的资源守卫；模型源码在其后从冻结根导入。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from word_ft_v3.resources import (DeviceLocks, double_probe, capture_snapshot,
                                     protected_path, check_idle)
    protected_path(out)
    if out.resolve() == source or source in out.resolve().parents:
        raise ValueError('不能在冻结源码内写诊断结果')
    with DeviceLocks(a.lock_root, a.expected_uuids) as locks:
        probe = double_probe([0, 1], 60000)
        if [x['uuid'] for x in probe['devices']] != a.expected_uuids:
            raise ValueError('两卡 UUID 与授权目标不符')
        out.mkdir(parents=True, exist_ok=False)
        def write(name, value):
            with (out / name).open('x') as f:
                json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.write('\n')
        write('admission.json', {'probe': probe, 'locks': locks.evidence,
              'script_sha256': sha(__file__), 'arguments': vars(a), 'pid': os.getpid(),
              'resource_helper_sha256': sha(Path(__file__).resolve().parents[1] / 'word_ft_v3/resources.py')})
        # CLI 的物理0/1准入与 torch 的逻辑0/1必须绑定同一 UUID。
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(a.expected_uuids)
        sys.path.insert(0, str(source))
        import torch
        import yaml
        import redimnet2
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if Path(redimnet2.__file__).resolve() != source / 'redimnet2/__init__.py':
            raise ValueError('模型未从冻结源导入')
        cfg = yaml.safe_load(Path(a.config).read_text())
        if cfg['model'] != 'ReDimNet2Custom' or cfg.get('compile_backbone'):
            raise ValueError('仅支持已审核的未编译 ReDimNet2Custom')
        if cfg['model_args']['spec_params'].get('do_spec_aug'):
            raise ValueError('本诊断不猜测丢失的 SpecAug RNG')
        payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if payload.get('epoch') != 47 or len(payload.get('model', {})) != 823:
            raise ValueError('本轮诊断仅接受已核 model47 的823项模型状态')
        raw = torch.load(a.wave, map_location='cpu', weights_only=True, mmap=True)
        wave = raw['wav']
        if tuple(wave.shape) != (96, 1, 32240) or wave.dtype != torch.float32 or not torch.isfinite(wave).all():
            raise ValueError('失败波形几何或有限性不符')
        state = {k[len('backbone.'):]: v for k, v in payload['model'].items() if k.startswith('backbone.')}
        def stats(t):
            if not torch.is_tensor(t):
                return None
            t = t.detach()
            finite = torch.isfinite(t)
            valid = t[finite]
            return {'shape': list(t.shape), 'dtype': str(t.dtype),
                    'nonfinite': int((~finite).sum()),
                    'max_abs_finite': float(valid.abs().max()) if valid.numel() else None}
        env = {'torch': torch.__version__, 'cuda': torch.version.cuda,
               'cudnn': torch.backends.cudnn.version(), 'source_import': str(redimnet2.__file__),
               'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
               'cudnn_tf32': torch.backends.cudnn.allow_tf32,
               'cudnn_benchmark': torch.backends.cudnn.benchmark,
               'cudnn_deterministic': torch.backends.cudnn.deterministic,
               'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
               'reconstructed_seed': cfg['seed'] + 4,
               'training': True, 'gradient_recording': True, 'optimizer_step': False,
               'failure_time_weights_and_rng_available': False, 'checkpoint_epoch': payload['epoch']}
        env['gradient_recording'] = not a.no_grad
        write('environment.json', env)
        ordinal = {'baseline-amp': 0, 'hooked-amp': 1, 'hooked-fp32': 2}[a.case]
        amp, hooks = a.case != 'hooked-fp32', a.case != 'baseline-amp'
        # 一个进程只做一个对照；下一次必须退出并取得全新空闲证明。
        for _single_case in range(1):
            # 每次使用同一指定 GPU，另一卡保留锁；无任何现有进程可被共享。
            if torch.cuda.is_initialized():
                raise ValueError('诊断前已有CUDA上下文，必须用独立新进程')
            check_idle(capture_snapshot(), [0, 1], 60000)
            torch.manual_seed(cfg['seed'] + 4)
            torch.cuda.manual_seed_all(cfg['seed'] + 4)
            model = redimnet2.ReDimNet2Custom(**cfg['model_args'])
            model.load_state_dict(state, strict=True)
            imports = {}
            for name, module in list(sys.modules.items()):
                if name == 'redimnet2' or name.startswith('redimnet2.'):
                    imports[name] = import_identity(name, module, source)
            write('imports.json', imports)
            model = model.cuda(0).train()
            if torch.cuda.device_count() != 2 or torch.cuda.current_device() != 0:
                raise ValueError('实际CUDA设备数或当前逻辑设备不符')
            x = wave.cuda(0)
            observations = []
            handles = []
            if hooks:
                def observe(name):
                    def callback(module, inputs, output):
                        outputs = output if isinstance(output, (list, tuple)) else (output,)
                        row = {'module': name, 'class': type(module).__name__,
                               'inputs': [stats(v) for v in inputs],
                               'outputs': [stats(v) for v in outputs]}
                        observations.append(row)
                    return callback
                for name, module in model.named_modules():
                    if name and (not list(module.children()) or name in ('spec', 'backbone', 'pool', 'bn', 'linear')):
                        handles.append(module.register_forward_hook(observe(name)))
            torch.cuda.reset_peak_memory_stats(0)
            started = time.monotonic()
            try:
                with torch.set_grad_enabled(not a.no_grad):
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                        embedding = model(x)
                torch.cuda.synchronize(0)
                write(f'case-{ordinal}.json', {'amp': amp, 'hooks': hooks,
                      'embedding': stats(embedding), 'seconds': time.monotonic() - started,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                      'observations': observations})
                del embedding
            except Exception as exc:
                write(f'case-{ordinal}-error.json', {'type': type(exc).__name__, 'message': str(exc),
                                                    'observations': observations})
                raise
            finally:
                for h in handles:
                    h.remove()
            del x, model
            torch.cuda.empty_cache()
        write('finished.json', {'cases': 1, 'case': a.case, 'optimizer_steps': 0,
                               'fresh_snapshot': capture_snapshot(), 'not_production_acceptance': True})


if __name__ == '__main__':
    main()
