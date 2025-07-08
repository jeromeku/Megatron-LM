#!/usr/bin/env python
# Show Megatron-LM TP×EP process-groups on a single CPU rank
import os

import torch.distributed as dist

from megatron.core.parallel_state import initialize_model_parallel
from megatron.core.process_groups_config import ModelCommProcessGroups

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")

dist.init_process_group("gloo", rank=0, world_size=1)
initialize_model_parallel(
    tensor_model_parallel_size=1,
    expert_model_parallel_size=1,   # EP off → groups are degenerate
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
)

pg = ModelCommProcessGroups()
breakpoint()
print("tp_group size:", dist.get_world_size(pg.tp))
print("ep_group size:", dist.get_world_size(pg.ep))
print("tp_ep_group size:", dist.get_world_size(pg.tp_ep))
print(f"{dist.get_world_size(pg.tp_ep_pp)=}")
print(f"{dist.get_world_size(pg.expt_tp)=}")
print(f"{dist.get_world_size(pg.tp_cp)=}")

dist.destroy_process_group()
