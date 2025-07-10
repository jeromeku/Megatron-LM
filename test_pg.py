import torch
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from megatron.core.parallel_state import initialize_model_parallel
from pathlib import Path
import megatron.core as mc
MEGATRON_ROOT = Path(mc.__file__).parent.resolve().as_posix()

from viztracer import VizTracer

tracer = VizTracer(include_files=[MEGATRON_ROOT], ignore_c_function=True, ignore_frozen=True, log_func_retval=True, log_func_args=True)

pp = 2
tp = 2
ep = 2

world_size = pp * tp * ep

fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)

tracer.output_file = "mpu.json"
# with tracer:
initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, expert_model_parallel_size=ep, create_gloo_process_groups=False)

# mesh = torch.distributed.device_mesh.init_device_mesh(
#     "cuda",
#     (world_size // 2, 2),
#     mesh_dim_names=(
#         "dp",
#         "tp",
#     ),
# )