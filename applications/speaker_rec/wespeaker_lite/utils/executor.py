# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tableprint as tp
import functools
import json
import math
import secrets
import traceback
import time
from pathlib import Path
import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from wespeaker_lite.utils import meters as tnt
from wespeaker_lite.utils import numerical_failure as forensic
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather

from torch import nn
from torch.distributions import Beta
from torch.nn.parameter import Parameter

import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device).to(X.dtype)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1).to(Y.dtype) * Y + (1 - coeffs.view(-1, 1).to(Y.dtype)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight

def to_ohe(num_classes, targets):
    # target_mask = embeds.new_zeros(embeds.size())
    target_mask = torch.zeros((targets.size()[0], num_classes),
                              dtype=torch.float,device=targets.device)
    target_mask.scatter_(1, targets.view(-1, 1).long(), 1.0)
    return target_mask

def _check_training_numerics(wav, embeds, loss, *, epoch, batch_index, rank,
                             optimizer, scaler, diagnostic_dir=None, utts=None,
                             model=None, targets=None, target_mask=None, lengths=None,
                             capture_context=None, original_batch=None, enable_amp=None,
                             absolute_iteration=None, epoch_iter=None):
    """One finite-path reduction; all ranks export before any fatal log/exit.

    The outer launcher treats fatal text as an immediate stop request. Never
    print that text while another rank is still writing its failure bundle.
    """
    local_checks = torch.stack([torch.isfinite(value.detach()).all()
                                for value in (wav, embeds, loss)])
    capture_required = capture_context is not None and capture_context.enabled
    capture_ok = not capture_required or capture_context.complete
    global_ok = (local_checks.all() & local_checks.new_tensor(capture_ok)).to(dtype=torch.int32)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        torch.distributed.all_reduce(global_ok, op=torch.distributed.ReduceOp.MIN)
    if bool(global_ok.item()):
        return

    flags = local_checks.cpu().tolist()
    world = torch.distributed.get_world_size() if distributed else 1
    location = {'epoch': int(epoch), 'batch': int(batch_index), 'rank': int(rank),
                'world_size': world, 'absolute_iteration': absolute_iteration,
                'epoch_iter': epoch_iter}
    artifact = None; evidence_error = None; attempted_directory = None
    if diagnostic_dir is not None:
        try:
            def number(value):
                value = float(value)
                return value if math.isfinite(value) else str(value)
            def describe(value):
                value = value.detach()
                return {'finite': bool(torch.isfinite(value).all().item()),
                        'dtype': str(value.dtype), 'shape': list(value.shape),
                        'max_abs': number(value.abs().max().item()) if value.numel() else None}
            input_wave = (capture_context.state['inputs']['wav']
                          if capture_context is not None and capture_context.state is not None else wav)
            summary = dict(location, wav=describe(input_wave), wav_after_forward=describe(wav),
                           embeds=describe(embeds), loss=describe(loss),
                           loss_value=number(loss.detach().item()), local_finite=flags,
                           pre_capture_ok=capture_ok,
                           learning_rates=[number(group['lr']) for group in optimizer.param_groups],
                           scaler_scale=number(scaler.get_scale()))
            if model is None:
                payload = {'schema': forensic.SCHEMA, 'location': location,
                           'wav': wav.detach().cpu().clone(), 'training_resume_checkpoint': False,
                           'replay_complete': False,
                           'replay_incomplete_reasons': ['legacy caller omitted model and pre-forward state']}
            else:
                payload = forensic.make_payload(model=model, optimizer=optimizer, scaler=scaler,
                    capture=capture_context, wav=wav, targets=targets, embeds=embeds, loss=loss,
                    target_mask=target_mask, lengths=lengths, utts=utts, original_batch=original_batch,
                    location=location, enable_amp=enable_amp)
            stem = (f'numerical_failure_epoch_{epoch:03d}_batch_{batch_index:06d}_rank_{rank}_'
                    f'{time.time_ns()}_{secrets.token_hex(8)}')
            attempted_directory = str(Path(diagnostic_dir) / stem)
            artifact = forensic.write_failure_bundle(diagnostic_dir, payload=payload,
                                                     summary=summary, stem=stem)
        except Exception:
            evidence_error = traceback.format_exc()
    else:
        evidence_error = 'diagnostic directory was not supplied; full evidence unavailable'
    status = dict(location, numerical_finite=all(flags), pre_capture_ok=capture_ok,
                  artifact=artifact, attempted_directory=attempted_directory, write_error=evidence_error)
    statuses = [status]; coordination_error = None
    if distributed:
        try:
            statuses = [None] * world
            torch.distributed.all_gather_object(statuses, status)
        except Exception:
            coordination_error = traceback.format_exc()
    coherent = (coordination_error is None and len(statuses) == world and all(
        isinstance(item, dict) and type(item.get('rank')) is int and item['rank'] == r
        and item.get('epoch') == epoch and item.get('batch') == batch_index
        and item.get('world_size') == world for r, item in enumerate(statuses)))
    coordination = {'schema': 'ptn-forward-failure-coordination/v1', 'location': location,
                    'participants': statuses, 'coherent': coherent,
                    'all_local_payloads_published': coherent and all(x['artifact'] is not None for x in statuses),
                    'all_rank_replay_complete': coherent and all(
                        x['artifact'] is not None and x['artifact']['replay_complete'] for x in statuses),
                    'coordination_error': coordination_error,
                    'before_current_batch_backward_or_optimizer': True}
    if artifact is not None:
        try:
            forensic.atomic_json_new(Path(artifact['directory']) / 'coordination.json', coordination)
        except Exception:
            evidence_error = (evidence_error or '') + traceback.format_exc()
    if distributed:
        try:
            torch.distributed.barrier()
        except Exception:
            coordination_error = (coordination_error or '') + traceback.format_exc()
    # Only now may torchrun's parent act on a fatal line: all export attempts are finished.
    failure_label = ('Non-finite training values' if any(
        isinstance(x, dict) and x.get('numerical_finite') is False for x in statuses)
        else 'Training evidence capture failure')
    print(f'{failure_label} stopped before backward: rank={rank}, evidence={artifact}, '
          f'evidence_write_error={evidence_error}, coordination_error={coordination_error}',
          file=sys.stderr, flush=True)
    raise FloatingPointError(
        f'Training numerical/evidence gate failed before backward: epoch={epoch}, '
        f'batch={batch_index}, rank={rank}, local_finite={flags}, pre_capture_ok={capture_ok}, '
        f'evidence={artifact}, evidence_write_error={evidence_error}, coordination_error={coordination_error}')


def run_epoch(dataloader,
              epoch_iter,
              model,
              criterion,
              optimizer,
              scheduler,
              margin_scheduler,
              epoch,
              rank,
              logger,
              scaler,
              mix_aug,
              enable_amp,
              log_batch_interval=100,
              device=torch.device('cuda'),
              diagnostic_dir=None):
    model.train()

    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    for i, batch in enumerate(tqdm(dataloader, total=epoch_iter,
                                   disable=(rank!=0), desc=f'Epoch: {epoch:03d}')):
        utts = batch['key']
        targets = batch['label']
        wav = batch['wav']
        # print(batch.keys())
        # features = batch['feat']

        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        # features = features.float().to(device)  # (B,T,F)
        wav = wav.float().to(device)  # (B,T,F)
        # print(f"wav : {wav.size()}")
        targets = targets.long().to(device)

        # Variable-length path: collate_varlen zero pads the batch and carries
        # the real sample count per utterance. Padded batches need those counts
        # so the front end and the pooling layer leave the padding out of their
        # statistics. Fixed-length batches have no `lengths` key and the model
        # is called exactly as before.
        lengths = batch.get('lengths', None)
        if lengths is not None:
            lengths = lengths.long().to(device)

        with torch.cuda.amp.autocast(enabled=enable_amp):
            target_mask = None
            if mix_aug is not None:
                num_classes = model.module.projection.out_features
                target_mask = to_ohe(num_classes, targets)
                if lengths is None:
                    wav, target_mask = mix_aug(wav, target_mask)
                else:
                    # Mixing two utterances of different true lengths is only
                    # meaningful on the padded view, so the mixture's real
                    # length is the longer of the two, not the padded width.
                    wav, target_mask = mix_aug(wav, target_mask)
                    perm_len = lengths[torch.randperm(lengths.size(0),
                                                      device=lengths.device)]
                    lengths = torch.maximum(lengths, perm_len)
                # targets = None

            # The bare-module pre-hook runs after DDP's buffer broadcast.
            with forensic.ForwardCapture(model, targets=targets, target_mask=target_mask,
                                         lengths=lengths, enabled=diagnostic_dir is not None) as capture_context:
                if lengths is None:
                    outputs = model(wav)  # (embed_a,embed_b) in most cases
                else:
                    outputs = model(wav, lengths=lengths)
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            # outputs = model.module.projection.float()(embeds.float(), targets)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model.module.projection(embeds.float(), targets, target_mask)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)

        try:
            _check_training_numerics(wav, embeds, loss, epoch=epoch,
                batch_index=i + 1, rank=rank, optimizer=optimizer, scaler=scaler,
                diagnostic_dir=diagnostic_dir, utts=utts, model=model, targets=targets,
                target_mask=target_mask, lengths=lengths, capture_context=capture_context,
                original_batch=batch, enable_amp=enable_amp,
                absolute_iteration=cur_iter, epoch_iter=epoch_iter)
        finally:
            capture_context.release()

        # loss, acc
        if (i + 1) % 16 == 0:
            loss_meter.add(loss.item())
            acc_meter.add(outputs.cpu().detach().float().numpy(), targets.cpu().numpy())

        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log
        if (i + 1) % log_batch_interval == 0:
            # loss_meter.add(loss.item())
            # acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())
            print(f"wav.size : {wav.size()}")
            logger.warning(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))

        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))

    # 数据侧丢弃了多少样本。
    #
    # 这些计数是**按进程**的：开了 num_workers 之后每个 worker 是独立进程，
    # 主进程这里通常一条都看不到（worker 自己会把前几条打到 stderr）。
    # 所以这里除了打印本进程的计数，还报出本 epoch 实际跑到了第几批，
    # 一旦和 epoch_iter 差很多，就能从主进程这一侧看出数据被大面积丢掉了。
    try:
        from wespeaker_lite.dataset import processor as _processor
        _processor.report_drop_counters(
            prefix="[dataset] dropped-sample summary")
        # 按语音活动边界取样那一步的计数：选了几段、切了几次、回退了几次。
        # 同样按进程统计，理由同上。没开这个功能时字典是空的，不打印任何东西。
        _processor.report_vad_stats()
    except Exception:
        pass
    if rank == 0:
        print(f"[dataset] 本 epoch 完成 {i + 1} / {epoch_iter} 个批次",
              file=sys.stderr, flush=True)

# https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
def EER(y, y_score):
    """
    :param y: True binary labels in range {0, 1} or {-1, 1}. If labels are not binary, pos_label should be explicitly given
    :param y_score: Target scores, can either be probability estimates of the positive class, confidence values,
    or non-thresholded measure of decisions (as returned by â€œdecision_functionâ€ on some classifiers).
    :return: eer, thresh
    """
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh.item()

def get_lr(opt):
    for g in opt.param_groups:
        return g['lr']

def get_wd(opt):
    for g in opt.param_groups:
        return g['weight_decay']

def wraped_all_gather(tensor,group=None):
    # torch.distributed.nn.functional.all_gather is shape-sensitive: the
    # collective matches on the flight size, so every rank must pass a tensor
    # with an identical shape or the ranks block on each other forever (NCCL
    # has no timeout on a mismatched size). Both call sites below are safe
    # today: the validation loaders use torch's DistributedSampler, which pads
    # each replica to the same length, so the per-rank batch shapes agree. The
    # invariant breaks if the validation set has fewer trials than ranks (a
    # rank gets an empty batch and the shapes diverge), so keep the trial count
    # above world_size when changing the validation protocol.
    with torch.no_grad():
        gathered_tensors = all_gather(tensor, group)
    return torch.stack(gathered_tensors)

def eval_epoch(
    validation_dataloader,
    model,
    optimizer,
    enable_amp,
    epoch,
    rank,
    log_df,
    exp_dir,
    val_ds_name="vox1",
    device=torch.device('cuda')
):
    model.eval()

    predicted_batches = []

    with torch.no_grad():
        for batch in tqdm(validation_dataloader, disable=(rank!=0), desc='validating'):
            (x0,x1),labels = batch

            labels = labels.to(device)
            x0 = x0.to(device)
            x1 = x1.to(device)

            with torch.cuda.amp.autocast(enabled=enable_amp):
                embed0 = model(x0)
                embed1 = model(x1)

            embeds = torch.cat([embed0,embed1],dim=0)
            embeds = F.normalize(embeds, p=2, dim=1)

            embeds = wraped_all_gather(embeds)
            labels = wraped_all_gather(labels)

            if rank == 0:
                predicted_batches.append((embeds.cpu(),labels.cpu()))

            del embeds, labels

    # Perform validation
    if rank == 0:
        new_row = {
            "epoch" : epoch,
            "lr" : get_lr(optimizer),
            'wd' : optimizer.param_groups[0]['weight_decay'],
            "margin" : model.module.projection.margin
        }

        labels = []
        embeddings = []

        for batch_embeddings, batch_labels in predicted_batches:
            if len(batch_labels.size()) == 2:
                for rank in range(batch_labels.size()[0]):
                    labels.append(batch_labels[rank].numpy())
                    embeddings.append(batch_embeddings[rank].numpy()[None,...])
            if len(batch_labels.size()) == 1:
                labels.append(batch_labels.numpy())
                embeddings.append(batch_embeddings.numpy()[None,...])

        labels = np.concatenate(labels,axis=0)
        embeddings = np.concatenate(embeddings,axis=0)

        print(f"total validation embeddings.shape : {embeddings.shape}")
        print(f"total validation labels.shape : {labels.shape}")

        assert len(labels) == len(embeddings)
        emb_dict = {}
        for i in range(len(labels)):
            emb_dict[labels[i].item()] = embeddings[i]

        scr0 = []
        scr1 = []
        # scr_avg = []
        protocol = validation_dataloader.dataset.vox1_protocol.copy()
        for _, row in protocol.iterrows():
            scr0.append(np.mean(cos_sim_mat(emb_dict[row.enroll_ind][:1],emb_dict[row.verify_ind][:1])).item())
            scr1.append(np.mean(cos_sim_mat(emb_dict[row.enroll_ind][1:],emb_dict[row.verify_ind][1:])).item())

        protocol['scr0'] = scr0
        protocol['scr1'] = scr1
        protocol['scr_avg'] = (protocol['scr0']+protocol['scr1'])/2

        # eer, _ = EER(protocol.label.values,protocol.score.values)
        # Full eval path: Vox1TestEvalDataset hands back 5 chunks of 3 s each
        # (chunk_len_sec=3.0, num_chunks=5). `scr0` averages the first chunk of
        # every chunk pair, i.e. one 3-second crop per utterance -- it is not a
        # full-length number either, so it is named after what it measures.
        new_row[f"{val_ds_name}-eer-3s"] = np.around(EER(protocol.label.values,protocol.scr0.values)[0]*100,3)
        new_row[f"{val_ds_name}-eer-tta"] = np.around(EER(protocol.label.values,protocol.scr1.values)[0]*100,3)
        new_row[f"{val_ds_name}-eer-avg"] = np.around(EER(protocol.label.values,protocol.scr_avg.values)[0]*100,3)

        # csv_out_path = os.path.join(exp_dir, "val_log.csv")
        # log_df = log_df.append(new_row, ignore_index=True)
        # log_df.to_csv(csv_out_path, index=False)
        # return log_df
        return new_row

def eval_epoch_simple(
    validation_dataloader,
    model,
    optimizer,
    enable_amp,
    epoch,
    rank,
    log_df,
    exp_dir,
    val_ds_name="vox1",
    device=torch.device('cuda')
):
    model.eval()

    predicted_batches = []

    with torch.no_grad():
        for batch in tqdm(validation_dataloader, disable=(rank!=0), desc='validating'):
            x,labels = batch

            # print(f"x : {x.size()}, labels : {labels.size()}")

            labels = labels.to(device)
            x = x.to(device)

            with torch.cuda.amp.autocast(enabled=enable_amp):
                embeds = model(x)

            embeds = F.normalize(embeds, p=2, dim=1)

            embeds = wraped_all_gather(embeds)
            labels = wraped_all_gather(labels)

            if rank == 0:
                predicted_batches.append((embeds.cpu(),labels.cpu()))

            del embeds, labels

    # Perform validation
    if rank == 0:
        new_row = {
            "epoch" : epoch,
            "lr" : get_lr(optimizer),
            "wd" : get_wd(optimizer),
            "margin" : model.module.projection.margin
        }

        labels = []
        embeddings = []

        for batch_embeddings, batch_labels in predicted_batches:
            # print(f"batch_embeddings : {batch_embeddings.size()}, batch_labels : {batch_labels.size()}")
            if len(batch_labels.size()) == 3:
                for rank in range(batch_labels.size()[0]):
                    labels.append(batch_labels[rank].numpy())
                    embeddings.append(batch_embeddings[rank].numpy())
            if len(batch_labels.size()) == 2:
                labels.append(batch_labels.numpy())
                embeddings.append(batch_embeddings.numpy())

        labels = np.concatenate(labels,axis=0)
        embeddings = np.concatenate(embeddings,axis=0)

        print(f"total validation embeddings.shape : {embeddings.shape}")
        print(f"total validation labels.shape : {labels.shape}")

        assert len(labels) == len(embeddings)
        emb_dict = {}
        for i in range(len(labels)):
            emb_dict[labels[i].item()] = embeddings[i]

        scr0 = []
        protocol = validation_dataloader.dataset.vox1_protocol.copy()
        for _, row in protocol.iterrows():
            scr0.append(np.mean(cos_sim_mat(emb_dict[row.enroll_ind][None,:],emb_dict[row.verify_ind][None,:])).item())
            # scr1.append(np.mean(cos_sim_mat(emb_dict[row.enroll_ind][1:],emb_dict[row.verify_ind][1:])).item())

        protocol['scr0'] = scr0
        # protocol['scr1'] = scr1
        # protocol['scr_avg'] = (protocol['scr0']+protocol['scr1'])/2

        # eer, _ = EER(protocol.label.values,protocol.score.values)
        # The dataset here is Vox1TestEvalDatasetSimple with start_time=0.0 and
        # duration=configs['eval_dur'] (8.0 s), so this number is a fixed
        # first-8-seconds window, NOT the full utterance. It used to be reported
        # as `-eer-fl` ("full length"), which was simply wrong about what it
        # measured; `-eer-8s` says what it is.
        try:
            new_row[f"{val_ds_name}-eer-8s"] = np.around(EER(protocol.label.values,protocol.scr0.values)[0]*100,3)
        except:
            new_row[f"{val_ds_name}-eer-8s"] = -1
        # new_row[f"vox1-eer-tta"] = np.around(EER(protocol.label.values,protocol.scr1.values)[0]*100,3)
        # new_row[f"vox1-eer-avg"] = np.around(EER(protocol.label.values,protocol.scr_avg.values)[0]*100,3)

        # csv_out_path = os.path.join(exp_dir, "val_log.csv")
        # log_df = log_df.append(new_row, ignore_index=True)
        # log_df.to_csv(csv_out_path, index=False)
        # return log_df
        return new_row

def cos_sim_mat(x0,x1):
    """
    x0 shape : (N,D)
    x1 shape : (M,D)
    D - dimensionality of embeddings
    """
    x0n = x0 / (np.linalg.norm(x0,axis=-1,keepdims=True)+1e-8)
    x1n = x1 / (np.linalg.norm(x1,axis=-1,keepdims=True)+1e-8)
    return np.sum(x0n[:,None,:]*x1n[None,:,:],axis=-1)


def set_bn_eval(m,flag=False):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.track_running_stats = flag