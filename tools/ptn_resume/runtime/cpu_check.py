"""真实model47 CPU完整恢复；不前向、不建训练输出、不初始化CUDA。"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time
import strict_restore as S


def run():
    S.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preflight must hide CUDA')
    S.verify_bundle(); source = S.verify_source(); dependencies = S.verify_dependencies()
    import torch
    import yaml
    torch.set_num_threads(1)
    S.need(not torch.cuda.is_initialized(), 'CPU check inherited CUDA context')
    from types import SimpleNamespace
    import decoder_runtime
    import audio_review
    decoder = decoder_runtime.audit_loaded_decoder(SimpleNamespace(decoder_library=str(S.DECODER)))
    legacy = Path(S.CONFIG['legacy']['bundle_root'])
    pins = json.loads((legacy / 'bundle_files.json').read_text())
    S.need(S.sha(legacy / 'audio_selfcheck.json') == pins['audio_selfcheck.json'], 'audio selfcheck changed')
    audio = audio_review.audit_audio_files(json.loads((legacy / 'audio_selfcheck.json').read_text()))
    sys.path[:0] = [str(S.SOURCE), str(S.SOURCE / 'applications/speaker_rec')]
    from wespeaker_lite.bin import train as T
    config = yaml.safe_load(Path(S.CONFIG['resume_config_path']).read_bytes()); S.validate_config(config)
    S.need(not S.RUN.exists(), 'new training output already exists')
    payload = S.checkpoint(); before = S.snapshot(S.CK); started = time.monotonic(); stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        model = T.SpeakerModel(T.get_speaker_model(config['model'])(**config['model_args']),
                               T.get_projection(config['projection_args']))
        model.load_state_dict(payload['model'], strict=True)
        optimizer = getattr(torch.optim, config['optimizer'])([
            {'params': model.backbone.parameters(), 'weight_decay': config['optimizer_args']['weight_decay']},
            {'params': model.projection.parameters(), 'weight_decay': config['loss_weight_decay_mult'] * config['optimizer_args']['weight_decay']}],
            **config['optimizer_args'])
        scheduler = getattr(T.schedulers, config['scheduler'])(optimizer, **config['scheduler_args'])
        margin = getattr(T.schedulers, config['margin_scheduler'])(model=model, **config['margin_update'])
        optimizer.load_state_dict(payload['optimizer'])
        scaler = torch.amp.GradScaler('cpu', enabled=True); scaler.load_state_dict(payload['scaler'])
        proof = S.verify_restored(model, optimizer, scaler, 48, 15001, 8, config, S.CONFIG['speaker_map']['path'],
                                  scheduler=scheduler, margin=margin)
        proof['imports'] = S.verify_imports()
    S.need(before == S.snapshot(S.CK) and not S.RUN.exists(), 'checkpoint changed or training output appeared')
    S.need(not torch.cuda.is_initialized() and not torch.distributed.is_initialized(), 'CPU gate initialized CUDA or communication')
    return {'schema': 'ptn47-CPU-real-restore/v1', 'cpu_only': True, 'source': source,
            'dependencies': dependencies, 'decoder': decoder, 'audio': audio,
            'strict_restore': proof, 'model_forward_or_update': False,
            'new_run_created': False, 'CUDA_initialized': False, 'seconds': time.monotonic() - started,
            'constructor_stdout': stdout.getvalue()}
