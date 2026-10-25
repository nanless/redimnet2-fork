#!/usr/bin/env python3
"""新接续的单rank入口；所有rank完整恢复后才解除前向/优化门。"""
import argparse
import datetime
import functools
import importlib
import json
import os
import fcntl
from pathlib import Path
import sys
import traceback
sys.dont_write_bytecode = True
import strict_restore as S


def main():
    p = argparse.ArgumentParser(); p.add_argument('--contract', required=True)
    p.add_argument('--contract-sha', required=True); p.add_argument('--package-sha', required=True)
    a = p.parse_args(); S.configure(a.contract, a.contract_sha, a.package_sha)
    return run_worker()


def run_worker():
    handles = []
    try:
        for path in [S.CONFIG['logical_chain_lock'], *S.CONFIG['required_existing_locks']]:
            path = Path(path); S.need(path.is_file() and not path.is_symlink(), 'stable worker lock missing')
            stream = path.open('rb'); fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB); handles.append(stream)
        rank=int(os.environ['RANK']);S.need(0<=rank<8 and int(os.environ['WORLD_SIZE'])==8 and int(os.environ['LOCAL_RANK'])==rank,'single-node W8 required')
        import entry as E
        handles.append(E.worker_ready_before_cuda(rank))
        from torch.distributed.elastic.multiprocessing.errors import record
        return record(admitted_worker)(handles,rank)
    finally:
        for stream in handles:stream.close()

def admitted_worker(handles,rank):
    sys.path[:0] = [str(S.SOURCE), str(S.SOURCE / 'applications/speaker_rec')]
    import torch
    import torch.distributed as dist
    rank = int(os.environ['RANK'])
    S.need(int(os.environ['WORLD_SIZE']) == 8 and int(os.environ['LOCAL_RANK']) == rank, 'single-node W8 required')
    from types import SimpleNamespace
    import decoder_runtime
    decoder = decoder_runtime.audit_loaded_decoder(SimpleNamespace(decoder_library=str(S.DECODER)))
    flags = {'ready': False, 'updated': False}
    def deny_forward(module, args):
        S.need(flags['ready'], 'model forward before all-rank exact restoration')
    hook = torch.nn.modules.module.register_module_forward_pre_hook(deny_forward)
    original_step = torch.optim.SGD.step
    @functools.wraps(original_step)
    def guarded_step(optimizer, *args, **kwargs):
        S.need(flags['ready'], 'optimizer step before all-rank exact restoration')
        result = original_step(optimizer, *args, **kwargs)
        if not flags['updated']:
            flags['updated'] = True
            receipt = {'rank': rank, 'epoch': flags['iteration'] // 15001 + 1, 'absolute_iteration': flags['iteration'],
                       'optimizer_step_completed': True, 'contract_sha256': S.CONTRACT_SHA,
                       'checkpoint_sha256': S.CK_SHA, 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            with (S.RUN / 'checks' / f'first_update_rank{rank}.json').open('x') as f:
                json.dump(receipt, f, indent=2)
            print('STRICT_FIRST_UPDATE ' + json.dumps(receipt), flush=True)
        return result
    torch.optim.SGD.step = guarded_step
    train = importlib.import_module('wespeaker_lite.bin.train'); original_loop = train.training_loop
    def strict_loop(start_epoch, configs, train_dataset, train_dataloader, epoch_iter, ddp_model, model,
                    criterion, optimizer, scaler, scheduler, margin_scheduler, rank, world_size, gpu,
                    logger, log_df, val_dls_dict, val_type, model_dir):
        error = None; proof = None
        try:
            S.verify_bundle()
            proof = S.verify_restored(model, optimizer, scaler, start_epoch, epoch_iter, world_size,
                configs, S.RUN / 'ptn/spk2id.json', scheduler=scheduler, margin=margin_scheduler)
            proof.update(rank=rank, world_size=world_size, imports=S.verify_imports(), decoder=decoder,
                         device=str(next(model.parameters()).device),
                         torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        except Exception:
            error = traceback.format_exc()
        consensus = [None] * 8
        dist.all_gather_object(consensus, {'rank': rank, 'error': error, 'exact': bool(proof and proof.get('model_exact'))})
        S.need(all(row and row['rank'] == j and row['error'] is None and row['exact'] for j, row in enumerate(consensus)),
               'all-rank restoration refused: ' + json.dumps(consensus))
        with (S.RUN / 'checks' / f'restore_verified_rank{rank}.json').open('x') as f:
            json.dump(proof, f, ensure_ascii=False, indent=2)
        dist.barrier(device_ids=[gpu]); flags['ready'] = True; hook.remove()
        print('STRICT_RESTORE_VERIFIED ' + json.dumps({'rank': rank, 'epoch': 48, 'absolute_iteration': 705047}), flush=True)
        original_scheduler_step = scheduler.step
        def tracked(current_iter):
            flags['iteration'] = current_iter
            result = original_scheduler_step(current_iter)
            if current_iter == 705047:
                with (S.RUN / 'checks' / f'first_iteration_rank{rank}.json').open('x') as f:
                    json.dump({'rank': rank, 'absolute_iteration': current_iter, 'lr': scheduler.get_lr(),
                               'contract_sha256': S.CONTRACT_SHA, 'before_first_forward': True}, f, indent=2)
            return result
        scheduler.step = tracked
        from epoch_guard import guarded_run_epoch
        original_run = train.run_epoch
        def guarded_epoch(*args, **kwargs):
            result = guarded_run_epoch(original_run, *args, **kwargs)
            S.need(args[7] != 48 or flags['updated'], 'epoch48 had no real optimizer update; refuse save')
            return result
        train.run_epoch = guarded_epoch
        try:
            return original_loop(start_epoch, configs, train_dataset, train_dataloader, epoch_iter,
                ddp_model, model, criterion, optimizer, scaler, scheduler, margin_scheduler, rank,
                world_size, gpu, logger, log_df, val_dls_dict, val_type, model_dir)
        finally:
            train.run_epoch = original_run
    train.training_loop = strict_loop
    original_save = train.save_checkpoint
    def checked_save(*args, **kwargs):
        S.verify_bundle()
        return original_save(*args, **kwargs)
    train.save_checkpoint = checked_save
    try:
        train.train(config=S.CONFIG['resume_config_path'])
        S.need(flags['updated'], 'worker ended without a real optimizer update')
        if dist.is_initialized(): dist.barrier(); dist.destroy_process_group()
    finally:
        torch.optim.SGD.step = original_step


if __name__ == '__main__':
    main()
