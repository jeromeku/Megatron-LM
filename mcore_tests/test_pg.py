import torch
from torch.testing._internal.distributed.fake_pg import FakeStore

import torch.distributed as dist
from megatron.core.parallel_state import initialize_model_parallel, RankGenerator
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_expert_model_parallel_group,
    get_tensor_model_parallel_group,
    get_expert_tensor_and_model_parallel_group,
    get_expert_data_parallel_group,
    get_tensor_and_data_parallel_group,
)
import megatron.core.parallel_state as mpu


from pathlib import Path
import megatron.core as mc

from viztracer import VizTracer
from contextlib import nullcontext

import os

MEGATRON_ROOT = Path(mc.__file__).parent.resolve().as_posix()
SHOULD_TRACE = os.getenv("SHOULD_TRACE", "0") == "1"


class FakeTracer(nullcontext):
    def save(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def get_default_tracer_args():
    args = dict(
        include_files=[MEGATRON_ROOT],
        ignore_c_function=True,
        ignore_frozen=True,
        log_func_retval=True,
        log_func_args=True,
    )
    return args


def setup_tracer(**kwargs):
    if SHOULD_TRACE:
        if len(kwargs) == 0:
            kwargs = get_default_tracer_args()
        return VizTracer(**kwargs)
    else:
        return FakeTracer()


WORLD_SIZE = 16
INTRANODE_SIZE = 8
NUM_NODES = WORLD_SIZE // INTRANODE_SIZE

cp_size = 1
pp_size = 1
tp_size = INTRANODE_SIZE
ep_mp_size = 8
ep_tp_size = INTRANODE_SIZE // ep_mp_size

fake_store = FakeStore()
torch.distributed.init_process_group("fake", store=fake_store, rank=0, world_size=WORLD_SIZE)

tracer = setup_tracer()
if SHOULD_TRACE:
    tracer.output_file = "mpu.json"

DEFAULT_ORDER = "tp-cp-ep-dp-pp"

def get_expert_rank_generator(ep_tp, ep_mp, pp, world_size, order=DEFAULT_ORDER, rank_offset=0):
    ep_tp_mp_pp = ep_tp * ep_mp * pp
    assert world_size % ep_tp_mp_pp == 0

    ep_dp = world_size // ep_tp_mp_pp
    
    return RankGenerator(tp=ep_tp, ep=ep_mp, dp=ep_dp, pp=pp, cp=1, order=order, rank_offset=rank_offset)


def get_attention_rank_generator(tp, pp, cp, world_size, order=DEFAULT_ORDER, rank_offset=0):
    
    mp = tp * pp * cp
    assert world_size % mp == 0

    dp = world_size // mp

    return RankGenerator(tp=tp, dp=dp, pp=pp, cp=cp, ep=1, order=order, rank_offset=rank_offset)

ep_generator = get_expert_rank_generator(ep_mp=ep_mp_size, ep_tp=ep_tp_size, pp=pp_size, world_size=WORLD_SIZE)
attn_generator = get_attention_rank_generator(tp=tp_size, pp=pp_size, cp=cp_size, world_size=WORLD_SIZE)

with tracer:
    initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        expert_model_parallel_size=ep_mp_size,
        expert_tensor_parallel_size=ep_tp_size,
        context_parallel_size=cp_size,
        create_gloo_process_groups=False,
    )

def get_pg_ranks(pg: dist.ProcessGroup):
    return dist.get_process_group_ranks(pg)

def print_pg(pg: dist.ProcessGroup, name: str = None):
    
    name = name or pg.name()
        
    ranks = get_pg_ranks(pg)

    print(f"{name=}: {ranks=}")

def print_generator_ranks(token: str, generator: RankGenerator):
    ranks = generator.get_ranks(token)
    print(f"{token} ranks: {ranks}")

def check_pg(pg: dist.ProcessGroup, token: str, attn_generator: RankGenerator, expert_generator: RankGenerator):
    print_pg(pg, token)
    print_generator_ranks(token, attn_generator)
    print_generator_ranks(token, expert_generator)

from functools import partial

check_pg = partial(check_pg, attn_generator=attn_generator, expert_generator=ep_generator)


dp = get_data_parallel_group()
ep_dp = get_expert_data_parallel_group()
tp = get_tensor_model_parallel_group()
ep = get_expert_model_parallel_group()
tp_dp = get_tensor_and_data_parallel_group()
ep_mp = get_expert_tensor_and_model_parallel_group()

check_pg(dp, "dp")
print()
check_pg(ep, "ep")
print()
check_pg(tp, "tp")
print()
check_pg(ep_mp, "tp-ep")
