import functools  # Add this import for partial
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    lambda_auto_wrap_policy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    wrap,
)


@dataclass
class DistributedState:
    ddp: bool = False
    ddp_rank: int = 0
    ddp_local_rank: int = 0
    ddp_world_size: int = 1
    master_process: bool = True


ddp_state = DistributedState()


def init_ddp():
    global ddp_state

    # assuming nccl here
    if (
        int(os.environ.get("RANK", -1)) != -1
        and int(os.environ.get("LOCAL_RANK", -1)) != -1
    ):
        dist.init_process_group(backend="nccl")

        ddp_state.ddp = True
        ddp_state.ddp_rank = int(os.environ.get("RANK", 0))
        ddp_state.ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
        ddp_state.ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp_state.master_process = ddp_state.ddp_rank == 0


init_ddp()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_avail_and_initialized():
    return ddp_state.ddp


def get_data_sampler(dataset):
    if is_dist_avail_and_initialized():
        return torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        return None


def distributed_mean(metrics: List[float], device: torch.device) -> float:
    count = len(metrics)
    metric = np.sum(metrics)
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()
        metric = torch.tensor(metric, dtype=torch.float32, device=device)
        count = torch.tensor(count, dtype=torch.long, device=device)
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        value = metric.item() / count.item()
    else:
        value = float(metric) / count
    return value


def get_device():
    if is_dist_avail_and_initialized():
        return "cuda:{:d}".format(get_local_rank())
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_local_rank():
    return ddp_state.ddp_local_rank


def get_world_size():
    return ddp_state.ddp_world_size


def get_rank():
    return ddp_state.ddp_rank


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_default_fsdp_config():
    """Returns a default FSDP configuration."""
    return {
        "mixed_precision": True,
        "use_fp16": False,
        "sharding_strategy": "FULL_SHARD",
        "backward_prefetch": "BACKWARD_PRE",
        "cpu_offload": True,  # False,
        "min_num_params": 1e6,
    }


def get_mixed_precision_config(use_fp16=False):
    """Returns mixed precision config based on BF16 or FP16 choice."""
    dtype = torch.float16 if use_fp16 else torch.bfloat16
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


def get_sharding_strategy(strategy_name):
    """Converts string strategy name to ShardingStrategy enum."""
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    return strategy_map.get(strategy_name, ShardingStrategy.FULL_SHARD)


def get_backward_prefetch(prefetch_name):
    """Converts string prefetch name to BackwardPrefetch enum."""
    prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }
    return prefetch_map.get(prefetch_name)


def get_auto_wrap_policy(model, config):
    """Returns an auto wrap policy based on model type and config."""
    # Try to get transformer layer class for transformer-based models
    # NOTE : we need to somehow tell FSP at which granularity to wrap modules
    # if it's too coarse (e.g. you wrap the whole model), then `gather` will literally
    # gather all parameters in the model, which won't give any memory savings
    # if it's too fine, then you will have a lot of FSDP wrappers, which will
    # slow down the training

    transformer_cls = None
    for module_name, module in model.named_modules():
        if any(
            name in module.__class__.__name__.lower()
            for name in ["transformerlayer", "transformerblock", "gptmlplayer"]
        ):
            transformer_cls = module.__class__
            break

    if transformer_cls is not None:
        return transformer_auto_wrap_policy(transformer_layer_cls={transformer_cls})
    else:
        # Create a partial function that will later be called with the required arguments
        min_params = float(config.get("min_num_params", 1e6))
        return functools.partial(size_based_auto_wrap_policy, min_num_params=min_params)


def fsdp_save_model(model, save_path):
    """Saves FSDP model in a way that can be loaded by non-FSDP code."""
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        state_dict = model.state_dict()
        if is_main_process():
            torch.save(state_dict, save_path)

    if is_dist_avail_and_initialized():
        torch.distributed.barrier()


def fsdp_load_model(model, load_path):
    """Loads state dict into an FSDP model."""
    if os.path.exists(load_path):
        state_dict = torch.load(load_path, map_location="cpu")
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(state_dict)

    if is_dist_avail_and_initialized():
        torch.distributed.barrier()


def wrap_model_with_fsdp(model, fsdp_config=None):
    """Wraps a model with FSDP using the specified config."""

    if False:  # not is_dist_avail_and_initialized():
        return model

    if fsdp_config is None:
        fsdp_config = get_default_fsdp_config()

    mixed_precision = None
    if fsdp_config.get("mixed_precision", False):
        mixed_precision = get_mixed_precision_config(fsdp_config.get("use_fp16", False))
        # Determine the target dtype based on mixed precision config
        target_dtype = (
            torch.float16 if fsdp_config.get("use_fp16", False) else torch.bfloat16
        )

    sharding_strategy = get_sharding_strategy(
        fsdp_config.get("sharding_strategy", "FULL_SHARD")
    )
    backward_prefetch = get_backward_prefetch(
        fsdp_config.get("backward_prefetch", "BACKWARD_PRE")
    )

    cpu_offload = None
    if fsdp_config.get("cpu_offload", False):
        cpu_offload = CPUOffload(offload_params=True)

    auto_wrap_policy = fsdp_config.get(
        "auto_wrap_policy", get_auto_wrap_policy(model, fsdp_config)
    )

    # Make sure to skip all lora related parameters
    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name:
            print(f"skipping lora param {name}")
            lora_params.append(param)

    # Add use_orig_params=True to allow parameters with different requires_grad settings
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=sharding_strategy,
        backward_prefetch=backward_prefetch,
        cpu_offload=cpu_offload,
        device_id=get_local_rank(),
        ignored_states=lora_params,
        # use_orig_params=True,  # Enable to handle mixed requires_grad parameters
    )

    return fsdp_model


def get_lora_ignored_fsdp_config(base_config=None):
    """
    Get FSDP config that ignores lora_a and lora_b parameters in the sharding,
    while using standard transformer wrapping policy for the rest.
    """
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )

    # Start with default config if not provided
    config = base_config or get_default_fsdp_config()

    def lambda_policy_fn(module):
        from mttl.models.modifiers.lora import LoRA

        if isinstance(module, LoRA):
            return True
        else:
            return (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

    lambda_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
    )

    # Add our custom policy to the config
    config["auto_wrap_policy"] = lambda_policy

    return config
