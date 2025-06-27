import random

import numpy as np
import torch
from torch import testing

import megatron.core.parallel_state as mpu
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.custom_fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import GradCommProcessGroups
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from pathlib import Path
import megatron

MEGATRON_ROOT = Path(megatron.core.__file__).parent.parent.resolve().as_posix()

# Test model for testing FSDP
class TestModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim * 4)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim * 4, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


# Test model with uniform shaped weights for testing FSDP
class TestModelUniform(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        return x


def setup_seed(seed):
    random.seed(seed)  # Set Python's built-in random seed
    np.random.seed(seed)  # Set NumPy's random seed
    torch.manual_seed(seed)  # Set PyTorch's CPU seed
    torch.cuda.manual_seed(seed)  # Set PyTorch's GPU seed (if using CUDA)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner for reproducibility


def setup_class():
    Utils.initialize_model_parallel()

def teardown_class():
    Utils.destroy_model_parallel()

def test_fsdp_with_process_groups(dp_size=1):
    """Test that FSDP works correctly with different process group configurations."""
    from torch.distributed.device_mesh import init_device_mesh
    from viztracer import VizTracer
    from pathlib import Path
    import os
    TEST_ROOT = Path(__file__).parent.resolve().as_posix()
    TRACE_DIR = 'traces'
    os.makedirs(TRACE_DIR, exist_ok=True)

    tracer = VizTracer(include_files=[MEGATRON_ROOT, TEST_ROOT], log_func_args=True, log_func_retval=True, ignore_c_function=True, ignore_frozen=True)
    print(f"Tracing dirs: {tracer.include_files}")
    # Skip test if we don't have enough GPUs
    world_size = torch.distributed.get_world_size()

    # Simple model config
    input_dim = 13
    output_dim = 17

    # Setup FSDP config - using optim_grads_params for full sharding test
    fsdp_config = DistributedDataParallelConfig(
        data_parallel_sharding_strategy="optim_grads_params",
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        bucket_size=10000,
        use_custom_fsdp=True,
    )

    # Create two identical models
    model1 = TestModel(input_dim=input_dim, output_dim=output_dim).cuda()

    transformer_config = TransformerConfig(
        num_attention_heads=1, num_layers=1, context_parallel_size=1  # Explicitly set CP=1
    )

    tracer.output_file = f"{TRACE_DIR}/fsdp.init.json"
    with tracer:
        fsdp_model1 = FullyShardedDataParallel(
            config=transformer_config,
            ddp_config=fsdp_config,
            module=model1,
            fsdp_unit_modules=[torch.nn.Linear],
        )

    # Create a 1D mesh with dimension [dp_size]
    device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
    tracer.output_file = f"{TRACE_DIR}/grad.pg.json"
    with tracer:
        grad_comm_pgs = GradCommProcessGroups()

    # Get dp process group from device mesh
    dp_group = device_mesh.get_group(mesh_dim="dp")
    grad_comm_pgs.dp = dp_group

    # Create optimizer config
    lr = 3
    optimizer_config = OptimizerConfig(optimizer="adam", lr=lr)
    grad_scaler = None

    tracer.output_file = f"{TRACE_DIR}/optimizer.init.json"
    with tracer:
        optimizer1 = DistributedOptimizer(
            optimizer=None,
            config=optimizer_config,
            grad_scaler=grad_scaler,
            init_state_fn=None,
            model_chunks=[fsdp_model1],
            per_model_buffers={0: [fsdp_model1.param_and_grad_buffer]},
            data_parallel_group=fsdp_model1.dp_cp_group,
            data_parallel_group_gloo=None,
            data_parallel_group_idx=0,
            distributed_optimizer_instance_id=0,
        )

    # Create identical inputs
    batch_size = 2
    input_data = torch.randint(0, 10, (batch_size, input_dim), device='cuda', dtype=torch.long)
    input_data = input_data.float()
    input_data.requires_grad = True

    def loss_fn(output, _):
        return output.sum()

    def train_step(model, optimizer, inputs):
        inputs_clone = inputs.clone().detach().requires_grad_(True)
        optimizer.zero_grad()
        outputs = model(inputs_clone)
        loss = loss_fn(outputs, None)
        loss.backward()
        optimizer.step()
        return outputs, loss

    tracer.output_file = f"{TRACE_DIR}/train_step.json"
    with tracer:
        out1, loss1 = train_step(fsdp_model1, optimizer1, input_data)

    # # Check parameters after optimization step
    # for (name1, param1), (_, param2) in zip(
    #     model1.named_parameters(), model2.named_parameters()
    # ):
    #     if hasattr(param1, 'fully_shard_param_local_shard') and hasattr(
    #         param2, 'fully_shard_param_local_shard'
    #     ):
    #         testing.assert_close(
    #             param1.fully_shard_param_local_shard,
    #             param2.fully_shard_param_local_shard,
    #             rtol=0,
    #             atol=0,
    #             msg=f"Parameters for {name1} don't match",
    #         )

    # if hasattr(torch.nn.parameter.Parameter, "main_grad"):
    #     # Custom fsdp adds the `main_grad` attribute function to the
    #     # torch Parameter, remove this attribute function so that
    #     # it doesn't conflict with the code in the non-custom fsdp
    #     # test branch.
    #     delattr(torch.nn.parameter.Parameter, "main_grad")

if __name__ == "__main__":
    setup_class()
    setup_seed(1234)
    test_fsdp_with_process_groups()
    