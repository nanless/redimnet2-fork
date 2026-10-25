"""保持普通DataLoader脚本入口；仅在rank父进程核每轮来源和192个worker。"""
import strict_restore as S


class CountedLoader:
    def __init__(self, loader):
        self.loader = loader
        self.count = 0
    def __len__(self): return len(self.loader)
    def __iter__(self):
        for batch in self.loader:
            self.count += 1
            yield batch


def guarded_run_epoch(original, *args, **kwargs):
    S.verify_bundle()
    S.need(S.sha(S.CK) == S.CK_SHA, 'model47 source changed')
    from loader_worker_audit import AuditedLoader
    import torch.distributed as dist
    loader, epoch, rank, world = args[0], args[7], args[8], dist.get_world_size()
    S.need(world == 8 and loader.num_workers == 24, 'formal data loader must remain W8×24')
    wrapped = AuditedLoader(loader, epoch, rank, world, S.RUN / 'checks/loader_workers')
    counted = CountedLoader(wrapped)
    result = original(counted, *args[1:], **kwargs)
    rows = [None] * world
    dist.all_gather_object(rows, {'rank': rank, 'epoch': epoch, 'batches': counted.count})
    S.need(all(row == {'rank': j, 'epoch': epoch, 'batches': 15001} for j, row in enumerate(rows)),
           'short/inconsistent epoch; refuse validation/checkpoint: ' + str(rows))
    from loader_worker_audit import atomic_new
    atomic_new(S.RUN / 'checks' / f'epoch_complete_epoch{epoch}_rank{rank}.json',
               {'rank': rank, 'epoch': epoch, 'batches_per_rank': 15001, 'ranks': rows,
                'contract_sha256': S.CONTRACT_SHA, 'before_validation_and_save': True})
    return result
