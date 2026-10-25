"""CPU-only topology and sampler contract; no implicit platform rank spawning."""
from __future__ import annotations
import math
from functools import lru_cache

SAMPLER_POLICY='balanced_cyclic_unique/v1'
TOPOLOGY_SCHEMA='single-node-cuda-topology/v3'

def topology(world):
    if type(world) is not int or world not in (2,8):raise ValueError('only single-node W2/W8 supported')
    return {'schema':TOPOLOGY_SCHEMA,'nodes':1,'gpus_per_node':world,'world_size':world,'backend':'nccl','launch_mode':'single_parent_torchrun'}

def validate(value):
    if not isinstance(value,dict) or value!=topology(value.get('world_size')) or any(type(value[k]) is not int for k in ('nodes','gpus_per_node','world_size')):raise ValueError('exact single-node W2/W8 topology required')
    return value['world_size']

def rank_environment(config,environment):
    world=validate(config['topology'])
    if int(environment.get('WORLD_SIZE','1'))!=world:raise ValueError('runtime/config world-size differs')
    rank=int(environment.get('RANK','-1'));local=int(environment.get('LOCAL_RANK','-1'))
    if not 0<=rank<world or local!=rank:raise ValueError('single-node rank/local_rank out of range')
    return world,rank,local

@lru_cache(maxsize=2)
def _permutation(count,cycle,seed):
    import numpy as np
    return np.random.default_rng(np.random.SeedSequence([seed,0xBACA,cycle])).permutation(count)

def step_indices(count,step,k,world,seed):
    """No dropped tails or duplicate families in a global step.

    Shuffle only at lcm(N, global_batch) positions. Within each such cycle,
    repeat one permutation; every N-position epoch covers all N families.
    This deliberately shares the ordering for G/gcd(N,G) epochs. It differs
    from v1's independently reshuffled epochs and is bound into v3 identity.
    """
    if any(type(x) is not int for x in (count,step,k,world,seed)) or count<1 or step<0 or k<1 or world not in (2,8) or seed<0:
        raise ValueError('invalid sampler geometry')
    size=k*world
    if count<size:raise ValueError('family pool smaller than complete global step')
    cycle_size=math.lcm(count,size)
    result=[]
    for position in range(step*size,(step+1)*size):
        cycle,offset=divmod(position,cycle_size)
        result.append(int(_permutation(count,cycle,seed)[offset%count]))
    if len(set(result))!=size:raise AssertionError('global family uniqueness violated')
    return result
