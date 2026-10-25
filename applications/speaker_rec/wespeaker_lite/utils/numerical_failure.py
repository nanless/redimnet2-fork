"""Failure-only state export; finite batches copy no parameters or optimizer.

Artifacts are private forensic inputs, never epoch checkpoints or permission to
resume. Snapshot the bare module after DDP's buffer broadcast. No GPU is touched
on import; RNG reads only devices already used by this model.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import AbstractContextManager
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import socket
import time
import traceback

import numpy as np
import torch

SCHEMA = 'ptn-forward-failure/v1'
MAX_SNAPSHOT_BYTES = 256 * 1024**2
IO_BYTES_PER_SECOND = 16 * 1024**2
_RUNTIME_FIELDS = ('p', 'dropout', 'inplace', 'margin', 'scale', 't', 'lanbuda',
                   'margin_type', 'dynamic_margin', 'cos_m', 'sin_m', 'th', 'mm', 'mmm')


def bare_model(model):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def _tree(value, *, cpu):
    if torch.is_tensor(value):
        tensor = value.detach()
        return tensor.to('cpu').clone() if cpu else tensor.clone()
    if isinstance(value, dict):
        result = OrderedDict() if isinstance(value, OrderedDict) else {}
        result.update((key, _tree(item, cpu=cpu)) for key, item in value.items())
        if hasattr(value, '_metadata'):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, (list, tuple)):
        return type(value)(_tree(item, cpu=cpu) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError('unsupported forensic value: ' + type(value).__name__)


def _tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(x) for x in value)
    return 0


def _devices(model):
    return sorted({tensor.device.index for tensor in list(model.parameters()) + list(model.buffers())
                   if tensor.device.type == 'cuda'})


def rng_state(model):
    numpy_state = np.random.get_state()
    devices = _devices(model)
    if devices and not torch.cuda.is_initialized():
        raise RuntimeError('model CUDA RNG requested before its existing context')
    return {'python': random.getstate(),
            'numpy': {'name': numpy_state[0], 'keys': numpy_state[1].tolist(),
                      'position': int(numpy_state[2]), 'has_gauss': int(numpy_state[3]),
                      'cached_gaussian': float(numpy_state[4])},
            'torch_cpu': torch.get_rng_state().clone(),
            'torch_cuda': {str(device): torch.cuda.get_rng_state(device).clone() for device in devices}}


def restore_rng(state, cuda_device_map=None):
    saved = {int(device) for device in state['torch_cuda']}
    mapping = {device: device for device in saved} if cuda_device_map is None else cuda_device_map
    if (any(type(x) is not int or x < 0 for x in mapping) or set(mapping) != saved
            or any(type(x) is not int or x < 0 for x in mapping.values())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError('explicit one-to-one CUDA RNG device map required')
    if saved and not torch.cuda.is_initialized():
        raise RuntimeError('replay must establish its own authorized model CUDA context first')
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value['name'], np.asarray(value['keys'], dtype=np.uint32),
                         value['position'], value['has_gauss'], value['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    for device, value in state['torch_cuda'].items():
        torch.cuda.set_rng_state(value, mapping[int(device)])


def _runtime(model):
    result = {}
    for name, module in model.named_modules():
        values = {'class': type(module).__module__ + '.' + type(module).__qualname__,
                  'training': module.training}
        for field in _RUNTIME_FIELDS:
            value = vars(module).get(field)
            if value is None or type(value) in (str, int, float, bool):
                if field in vars(module):
                    values[field] = value
        result[name] = values
    return result


def _versions(model):
    return {name: int(parameter._version) for name, parameter in model.named_parameters()}


class ForwardCapture(AbstractContextManager):
    """One bare-module call, post-DDP broadcast and post-mix; bounded device clones.

    A snapshot error is retained until the existing collective numeric gate so
    one rank cannot escape alone before its peers reach the same gate.
    """
    def __init__(self, model, *, targets, target_mask=None, lengths=None,
                 enabled=True, max_snapshot_bytes=MAX_SNAPSHOT_BYTES):
        self.model = bare_model(model)
        self.inputs = {'targets': targets, 'target_mask': target_mask, 'lengths': lengths}
        self.enabled = enabled
        self.limit = max_snapshot_bytes
        self.calls = 0
        self.state = None
        self.error = None
        self.handle = None

    def __enter__(self):
        if self.enabled:
            self.handle = self.model.register_forward_pre_hook(self._capture, with_kwargs=True)
        return self

    def _capture(self, module, args, kwargs):
        self.calls += 1
        if self.calls != 1:
            self.error = 'bare model invoked more than once for one capture'
            return
        try:
            if not args or not torch.is_tensor(args[0]):
                raise ValueError('positional waveform required for forensic capture')
            inputs = dict(self.inputs, wav=args[0], lengths=kwargs.get('lengths', self.inputs['lengths']))
            buffers = dict(module.named_buffers(remove_duplicate=False))
            count = _tensor_bytes(inputs) + _tensor_bytes(buffers)
            if type(self.limit) is not int or self.limit <= 0 or count > self.limit:
                raise ValueError('bounded forward snapshot byte budget exceeded')
            started = time.monotonic()
            self.state = {'rng': rng_state(module), 'buffers': _tree(buffers, cpu=False),
                          'inputs': _tree(inputs, cpu=False), 'parameter_versions': _versions(module),
                          'runtime': _runtime(module), 'snapshot_tensor_bytes': count,
                          'host_capture_seconds': time.monotonic() - started,
                          'position': 'bare_module_pre_forward_after_DDP_buffer_broadcast'}
        except Exception:
            self.state = None
            self.error = traceback.format_exc()

    @property
    def complete(self):
        return self.enabled and self.calls == 1 and self.state is not None and self.error is None

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False

    def release(self):
        self.state = None
        self.inputs = None


def runtime_identity():
    return {'torch': str(torch.__version__), 'numpy': str(np.__version__),
            'cuda_build': torch.version.cuda, 'hostname': socket.gethostname(),
            'pid': os.getpid(), 'process_group': os.getpgrp(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'environment': {key: os.environ.get(key) for key in (
                'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'CUDA_VISIBLE_DEVICES',
                'PTN_BUNDLE_MANIFEST_SHA', 'PTN_RESUME_NONCE', 'CUBLAS_WORKSPACE_CONFIG')}}


def make_payload(*, model, optimizer, scaler, capture, wav, targets, embeds, loss,
                 target_mask=None, lengths=None, utts=None, original_batch=None, location=None,
                 enable_amp=None):
    model = bare_model(model)
    post = _tree(model.state_dict(), cpu=True)
    post_buffers = dict(model.named_buffers(remove_duplicate=False))
    reasons = []
    if capture is None or not capture.complete:
        reasons.append('pre-forward snapshot unavailable or failed')
    pre = None
    state = capture.state if capture is not None else None
    if state is not None:
        if _versions(model) != state['parameter_versions']:
            reasons.append('parameter versions changed during forward')
        if set(post_buffers) != set(state['buffers']):
            reasons.append('registered buffer members changed during forward')
        elif any(post_buffers[name].shape != value.shape or post_buffers[name].dtype != value.dtype
                 for name, value in state['buffers'].items()):
            reasons.append('registered buffer shape/dtype changed during forward')
        if json.dumps(_runtime(model), sort_keys=True) != json.dumps(state['runtime'], sort_keys=True):
            reasons.append('known forward scalar attributes or train modes changed')
        pre = copy.copy(post)
        for name, value in state['buffers'].items():
            if name in pre:
                pre[name] = _tree(value, cpu=True)
    names = {id(value): name for name, value in model.named_parameters()}
    groups = [[names.get(id(value)) for value in group['params']] for group in optimizer.param_groups]
    if any(name is None for group in groups for name in group):
        reasons.append('optimizer parameter is outside captured model')
    prepared = state['inputs'] if state is not None else {
        'wav': wav, 'targets': targets, 'target_mask': target_mask, 'lengths': lengths}
    inputs = _tree(prepared, cpu=True)
    if not torch.is_tensor(inputs.get('wav')) or not torch.is_tensor(inputs.get('targets')):
        reasons.append('prepared waveform or targets missing')
    elif inputs['targets'].ndim != 1 or inputs['targets'].shape[0] != inputs['wav'].shape[0]:
        reasons.append('prepared waveform/target batch differs')
    original = {key: original_batch[key] for key in ('wav', 'label', 'key', 'lengths')
                if original_batch is not None and key in original_batch}
    return {'schema': SCHEMA, 'purpose': 'forensic_single_prepared_batch_replay_only',
            'training_resume_checkpoint': False, 'location': dict(location or {}),
            'runtime_identity': runtime_identity(), 'enable_amp': enable_amp,
            'model_post_forward': post, 'model_pre_forward': pre,
            'buffers_pre_forward': _tree(state['buffers'], cpu=True) if state is not None else None,
            'buffers_post_forward': _tree(post_buffers, cpu=True),
            'runtime_pre_forward': state['runtime'] if state is not None else None,
            'runtime_post_forward': _runtime(model),
            'rng_pre_forward': state['rng'] if state is not None else None,
            'rng_at_failure_capture': rng_state(model),
            'optimizer_state': _tree(optimizer.state_dict(), cpu=True),
            'optimizer_parameter_names': groups,
            'scaler_state': _tree(scaler.state_dict(), cpu=True),
            'scaler_enabled': bool(scaler.is_enabled()),
            'inputs': inputs, 'wav': inputs['wav'], 'targets': inputs['targets'],
            'utterance_keys': list(utts) if utts is not None else None,
            'original_batch': _tree(original, cpu=True),
            'original_batch_timing': 'read at failure; inputs is the authoritative pre-forward snapshot',
            'embeds_at_failure': _tree(embeds, cpu=True), 'loss_at_failure': _tree(loss, cpu=True),
            'pre_capture_error': capture.error if capture is not None else 'missing capture',
            'snapshot_tensor_bytes': state['snapshot_tensor_bytes'] if state is not None else None,
            'replay_complete': not reasons, 'replay_incomplete_reasons': reasons,
            'saved_before_current_batch_backward_or_optimizer': True,
            'limitations': ['not a DataLoader/sampler resume or epoch checkpoint',
                           'same audited source/config and numerical backend required',
                           'forward parameter .data mutation is unsupported; audited PTN does not do it']}


class _PacedWriter:
    def __init__(self, stream):
        self.stream = stream
        self.start = time.monotonic()
        self.bytes = 0
    def write(self, data):
        view = memoryview(data)
        for offset in range(0, len(view), 1024**2):
            chunk = view[offset:offset + 1024**2]
            if self.stream.write(chunk) != len(chunk):
                raise OSError('short forensic payload write')
            self.bytes += len(chunk)
            time.sleep(max(0., self.bytes / IO_BYTES_PER_SECOND - (time.monotonic() - self.start)))
        return len(view)
    def flush(self):
        return self.stream.flush()


def _digest(path):
    value = hashlib.sha256(); start = time.monotonic(); total = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block); total += len(block)
            time.sleep(max(0., total / IO_BYTES_PER_SECOND - (time.monotonic() - start)))
    return value.hexdigest()


def atomic_json_new(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)  # Atomic publication without replacing an old name.
    temporary.unlink()


def write_failure_bundle(directory, *, payload, summary, stem=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    location = payload['location']
    if stem is None:
        stem = ('numerical_failure_epoch_%03d_batch_%06d_rank_%d_%d_%s' % (
            location['epoch'], location['batch'], location['rank'], time.time_ns(), secrets.token_hex(8)))
    if Path(stem).name != stem or stem in ('', '.', '..'):
        raise ValueError('invalid forensic directory stem')
    event = directory / stem
    event.mkdir(mode=0o700, exist_ok=False)
    try:
        temporary = event/'payload.pt.part'
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, _PacedWriter(stream)); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, event/'payload.pt'); temporary.unlink()
        record = dict(summary, schema=SCHEMA, payload_file='payload.pt',
                      wav_file=str((event/'payload.pt').resolve()),
                      payload_sha256=_digest(event/'payload.pt'),
                      payload_bytes=(event/'payload.pt').stat().st_size,
                      replay_complete=payload['replay_complete'],
                      replay_incomplete_reasons=payload['replay_incomplete_reasons'])
        atomic_json_new(event/'complete.json', record)
        return {'directory': str(event), 'manifest': str(event/'complete.json'),
                'payload_sha256': record['payload_sha256'], 'replay_complete': record['replay_complete']}
    except Exception:
        try: atomic_json_new(event/'write_error.json', {'traceback': traceback.format_exc(), 'complete': False})
        except Exception: pass
        raise


def restore_pre_forward(model, payload, *, optimizer=None, scaler=None, cuda_device_map=None):
    """Explicit trusted forensic replay helper. Never performs forward or step."""
    if payload.get('schema') != SCHEMA or payload.get('replay_complete') is not True:
        raise ValueError('incomplete or wrong forensic schema')
    model = bare_model(model)
    modules = dict(model.named_modules())
    runtime = payload['runtime_pre_forward']
    if set(modules) != set(runtime):
        raise ValueError('replay module members differ')
    for name, values in runtime.items():
        if type(modules[name]).__module__ + '.' + type(modules[name]).__qualname__ != values['class']:
            raise ValueError('replay model class differs')
        for key, value in values.items():
            if key not in ('training', 'class') and vars(modules[name]).get(key) != value:
                raise ValueError('replay forward attribute differs: ' + name + '.' + key)
    model.load_state_dict(payload['model_pre_forward'], strict=True)
    buffers = dict(model.named_buffers(remove_duplicate=False))
    if set(buffers) != set(payload['buffers_pre_forward']):
        raise ValueError('replay buffer members differ')
    with torch.no_grad():
        for name, source in payload['buffers_pre_forward'].items():
            target = buffers[name]
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError('replay buffer shape/dtype differs: ' + name)
            target.copy_(source.to(target.device))
    for name, values in runtime.items():
        modules[name].training = values['training']
    if optimizer is not None:
        names = {id(value): name for name, value in model.named_parameters()}
        groups = [[names.get(id(value)) for value in group['params']] for group in optimizer.param_groups]
        if groups != payload['optimizer_parameter_names']:
            raise ValueError('replay optimizer parameter order differs')
        optimizer.load_state_dict(payload['optimizer_state'])
    if scaler is not None:
        if bool(scaler.is_enabled()) != payload['scaler_enabled']:
            raise ValueError('replay scaler enable policy differs')
        scaler.load_state_dict(payload['scaler_state'])
    mapping = ({int(x): int(x) for x in payload['rng_pre_forward']['torch_cuda']}
               if cuda_device_map is None else cuda_device_map)
    if set(mapping.values()) != set(_devices(model)):
        raise ValueError('replay CUDA RNG map does not match model devices')
    restore_rng(payload['rng_pre_forward'], mapping)
    return payload['inputs']
