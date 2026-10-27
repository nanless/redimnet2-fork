"""由已保存的八卡监控估算显存容量；不启动GPU，不把容量当提速实测。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(ok, text):
    if not ok: raise ValueError(text)


def estimate(snapshot, throughput_per_hour, remaining_sources, reserve_mib=8192, peak_factor=1.15):
    require(snapshot.get('schema') == 'annotation-existing-monitor-capacity/v1', 'wrong monitor summary schema')
    devices = snapshot['devices']
    require(len(devices) == 8 and len({g['uuid'] for g in devices}) == 8, 'eight distinct physical GPUs required')
    require(type(reserve_mib) is int and reserve_mib >= 0 and type(peak_factor) in (int,float) and
            math.isfinite(peak_factor) and peak_factor >= 1, 'invalid explicit memory safety assumptions')
    require(type(throughput_per_hour) in (int,float) and math.isfinite(throughput_per_hour) and throughput_per_hour > 0,
            'positive observed source throughput required')
    require(type(remaining_sources) is int and remaining_sources >= 0, 'invalid remaining source count')
    instance_mib, directly_observed_resident_mib, overhead_mib, totals, utils = [], [], [], [], []
    for gpu in devices:
        procs = gpu['compute_pids']
        require(len(procs) == 1 and procs[0]['gpu_uuid'] == gpu['uuid'], 'monitor must show one bound process per GPU')
        total, used, free, resident = (gpu['total_mib'], gpu['used_mib_latest'], gpu['free_mib_latest'], procs[0]['used_mib'])
        require(all(type(x) is int for x in (total,used,free,resident)) and 0 < resident <= used and
                total > 0 and free >= 0 and used + free <= total, 'inconsistent NVML memory accounting')
        low, high = gpu['used_mib_observed_min'], gpu['used_mib_observed_max']
        require(type(low) is int and type(high) is int and 0 <= low <= used <= high <= total,
                'invalid observed memory range')
        u = gpu['utilization_mean']
        require(type(u) in (int,float) and math.isfinite(u) and 0 <= u <= 100, 'invalid utilization')
        # total-used-free是驱动保留；used-resident是已观察到的非该实例占用。
        overhead_mib.append(total - free - resident)
        # 短窗出现的总used上升量也归入实例预算，避免只用末点较低值。
        instance_mib.append(resident + gpu['used_mib_observed_max'] - used)
        directly_observed_resident_mib.append(resident)
        totals.append(total);utils.append(u)
    instance, overhead, total = max(instance_mib), max(overhead_mib), min(totals)
    raw_limit = max(0, (total - overhead) // instance)
    budget = math.ceil(instance * peak_factor)
    guarded_limit = max(0, (total - overhead - reserve_mib) // budget)
    scenarios = [{'replicas_per_GPU': p, 'processes_across_8_GPU': p*8,
                  'resident_plus_overhead_mib_per_GPU': p*instance+overhead,
                  'remaining_memory_mib_per_GPU': total-overhead-p*instance,
                  'within_static_resident_limit': p <= raw_limit,
                  'within_explicit_guarded_budget': p <= guarded_limit,
                  'actual_speedup': None,
                  'linear_speedup_only_if_per_process_rate_unchanged': p}
                 for p in range(1,7)]
    return {'schema': 'annotation-eight-GPU-capacity-estimate/v1', 'monitor_observed_utc': snapshot['observed_utc'],
            'monitor_window_seconds': {'memory':snapshot['GPU_window_seconds'], 'utilization':snapshot['utilization_window_seconds']},
            'observed_single_process_mib_range': [min(directly_observed_resident_mib),max(directly_observed_resident_mib)],
            'instance_memory_proxy_mib_range': [min(instance_mib),max(instance_mib)],
            'instance_memory_proxy_policy':'latest process resident plus observed total-card used increase; not a directly measured process peak',
            'capacity_reference_instance_mib':instance, 'non_instance_or_unavailable_mib':overhead,
            'total_mib_reference':total, 'static_memory_only_replicas_per_GPU':raw_limit,
            'assumptions':{'extra_reserve_mib':reserve_mib,'per_instance_peak_factor':peak_factor},
            'guarded_memory_only_replicas_per_GPU':guarded_limit,
            'mean_GPU_busy_percent':sum(utils)/len(utils), 'memory_scenarios':scenarios,
            'measured_current_sources_per_hour':throughput_per_hour, 'remaining_sources_at_baseline':remaining_sources,
            'assumed_speedup_ETA': [{'assumed_speedup':s, 'conditional_sources_per_hour':s*throughput_per_hour,
                                    'conditional_days':remaining_sources/(s*throughput_per_hour*24)} for s in (1,1.5,2,2.5)],
            'speedup_measured':False, 'GPU_test_executed':False, 'extra_processes_started':0,
            'current_maintained_launcher_replicas_per_GPU':1, 'multi_replica_launcher_validated':False,
            'dispatch_or_CPU_memory_bandwidth_saturation_measured':False,
            'limitations':['利用率不是SM占用率；不从100/利用率推导必然提速或理论上限。',
                          '短窗显存不是历史峰值；静态容量与明确安全余量都不是启动许可。',
                          '多进程可能争用CUDA调度、CPU、内存及共享盘，线性速度列仅为假设。',
                          '现有启动器每UUID只准一个worker，增加每卡副本需另实现并审阅资源归属。',
                          'BF16、SDPA收益未实测；此计算不改变旧结果、配置或训练/评测资格。']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--snapshot-sha',required=True)
    p.add_argument('--throughput-per-hour',type=float,required=True);p.add_argument('--remaining-sources',type=int,required=True)
    p.add_argument('--reserve-mib',type=int,default=8192);p.add_argument('--peak-factor',type=float,default=1.15)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(all(not v.is_symlink() for v in (a.snapshot,*a.snapshot.absolute().parents)), 'snapshot symlink refused')
    raw=a.snapshot.read_bytes();require(hashlib.sha256(raw).hexdigest()==a.snapshot_sha,'snapshot SHA differs')
    out=a.output.absolute();require(not out.exists() and out.parent.is_dir() and
        all(not v.is_symlink() for v in (out,*out.parents)), 'new physical output required')
    require(out != a.snapshot.absolute(),'output overlaps input')
    report=estimate(json.loads(raw),a.throughput_per_hour,a.remaining_sources,a.reserve_mib,a.peak_factor)
    report['monitor_summary_sha256']=a.snapshot_sha
    with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
