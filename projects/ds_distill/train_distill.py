import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from tqdm import tqdm

from mttl.arguments import EvaluationConfig, ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.logging import logger, setup_logging
from mttl.models.expert_model import (
    MultiExpertModel,
    MultiExpertModelConfig,
    disable_modifiers,
    set_active_expert,
)
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.utils import MetricLogger, transfer_batch_to_device
from mttl.utils import remote_login

from projects.ds_distill.utils import (
    ExtendedLinear,
    ExtendedEmbedding,
    get_lora_injected_layers,
    create_batch,
    run_evaluation,
    reset_lora_params,
    silence_logger
)

@dataclass
class DSDistillConfig(ExpertConfig):
    tie_input_outputs: bool = False
    n_inner_steps: int = 1
    normalize_embeddings: bool = False
    prefix_length: int = 0
    n_samples: int = 10
    seq_len: int = 64


def get_grad_loss(lora_layers):
    """Calculate gradient-based loss metrics for LoRA layers.
    
    Computes cosine similarity loss, L1 loss, and L2 loss between the LoRA 
    matrices (A*B) and the gradients of the weight matrices.
    """
    sum_dotp = 0.0  # Sum of dot products between AB and gradients
    sum_ab_2 = 0.0  # Sum of squared AB values
    sum_grad_2 = 0.0  # Sum of squared gradient values
    sum_l1 = 0  # Sum of L1 distances
    sum_l2 = 0  # Sum of L2 distances
    
    for name, lora_layer in lora_layers.items():
        # Compute the effective LoRA weight matrix AB
        AB = (
            lora_layer.lora_b["oracle"].T @ lora_layer.lora_a["oracle"].T
        )  # (out_features, in_features)
        W_grad = lora_layer.weight.grad  # (out_features, in_features)
        
        # Accumulate dot product for cosine similarity calculation
        sum_dotp += (AB * W_grad).sum()
        sum_ab_2 += (AB * AB).sum()
        sum_grad_2 += (W_grad * W_grad).sum()
        
        # Calculate L1 and L2 distances between AB and gradients
        sum_l1 += (AB - W_grad).abs().sum()
        sum_l2 += (AB - W_grad).pow(2).sum()
    
    # Average L1 and L2 losses across all layers
    sum_l1 = sum_l1 / len(lora_layers)
    sum_l2 = sum_l2 / len(lora_layers)

    # Calculate cosine similarity and return 1 - cos_sim as loss
    cos_sim = sum_dotp / (sum_ab_2 * sum_grad_2).sqrt()
    return 1 - cos_sim, sum_l1, sum_l2

def train_and_eval(model, eval_dataloader):
    """Train a fast expert for a few steps and evaluate its performance.
    
    This function resets the fast expert's LoRA parameters, trains it for a few
    optimization steps, and logs the evaluation loss at each step to track improvement.
    """
    fast_expert = model.get_expert_instance("fast_expert")
    reset_lora_params(fast_expert)  # Reset parameters to initial state

    # Get trainable parameters for the fast expert
    fast_expert_params = [
        param for name, param in model.named_parameters() if "fast_expert" in name
    ]
    for f_pam in fast_expert_params:
        f_pam.requires_grad = True
    optim = torch.optim.Adam(fast_expert_params, lr=5e-5)

    # Configure training arguments for inner optimization
    args.trainable_param_names = ".*fast_expert.*"
    args.learning_rate = 5e-3
    args.total_steps = 5

    with silence_logger():
        (optim, scheduler), trainable_param_names = get_optimizer_and_scheduler(
            model, args, -1
        )

    with set_active_expert(model, "fast_expert"):
        # Evaluate initial performance before training
        loss = run_evaluation(model, eval_dataloader)
        logger.info(f"\tEvaluation loss: {loss} before training")

        # Inner training loop
        for it in range(args.total_steps):
            # Create training batch and move to device
            batch = create_batch(args, labels=True)
            batch = transfer_batch_to_device(batch, model.device)
            
            # Forward pass and optimization step
            outputs = model(**batch)
            optim.zero_grad()
            outputs.loss.backward()
            optim.step()
            scheduler.step()
            
            # Evaluate after each step to track progress
            loss = run_evaluation(model, eval_dataloader)
            logger.info(
                f"\tInner Evaluation loss: {loss} at step {it} lr {optim.param_groups[0]['lr']}"
            )

        model.zero_grad()


def ds_distill(args: EvaluationConfig):
    """Main function for dataset distillation using LoRA experts.
    
    This function implements a dataset distillation approach where learnable embeddings
    are optimized to encode task-specific information that allows a fast expert to 
    quickly adapt to the same task as an oracle expert. The process involves:
    1. Loading an oracle expert from a library
    2. Creating learnable input/output embeddings
    3. Training these embeddings to minimize the difference between oracle gradients
       and fast expert LoRA matrices
    """

    seed_everything(args.seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging and metrics tracking
    setup_logging(args.output_dir)
    logger.info("Args: {}".format(args.to_json()))
    metric_logger = MetricLogger()

    remote_login(args.remote_token)

    # Ensure single expert selection for distillation
    assert (
        args.expert_name or len(args.finetune_task_name.split(",")) == 1
    ), "Please provide a single expert selection for which to extract info"
    expert_name = args.expert_name or args.finetune_task_name.split(",")[0]

    # Load the oracle expert from the library
    library = ExpertLibrary.get_expert_library(
        repo_id=args.library_id,
        token=args.remote_token,
        destination_id=args.destination_library_id,
        selection=expert_name,
    )
    expert = library[expert_name]
    train_cfg = ExpertConfig.from_dict(expert.training_config)

    # Override specific configuration arguments if provided
    for arg in ["subsample_test", "predict_batch_size", "model"]:
        if hasattr(args, arg) and getattr(args, arg) is not None:
            logger.info(f"Overriding {arg} with {getattr(args, arg)}")
            setattr(train_cfg, arg, getattr(args, arg))

    base_model = train_cfg.model

    # Configure model loading with optimization settings
    loading_kwargs = {
        "device_map": args.device_map,
        "precision": "bf16",
        "attn_implementation": "flash_attention_2",
    }

    # Initialize multi-expert model with base model
    model = MultiExpertModel(
        MultiExpertModelConfig(
            base_model=base_model,
        ),
        **loading_kwargs,
    )

    # Add oracle expert (ground truth) and fast expert (to be trained)
    model.add_expert_instance(expert, expert_name="oracle")
    model.add_empty_expert(
        expert_name="fast_expert", expert_config=expert.expert_config
    )

    # Freeze all model parameters initially
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Get LoRA layers for gradient-based training
    lora_layers = get_lora_injected_layers(model)

    # Build datamodule using the same configuration as the oracle expert
    dm = get_datamodule(train_cfg)

    # Calculate dimensions for learnable embeddings
    args.N_NEW_TOKENS = args.n_samples * args.seq_len
    old_embeds = model.model.get_input_embeddings()
    old_unembeds = model.model.get_output_embeddings()
    args.OFFSET = old_embeds.num_embeddings  # Offset for new token indices
    logger.info(f'Added {args.N_NEW_TOKENS} new tokens to the model')

    # Create learnable input embeddings by copying and shuffling existing embeddings
    learnable_E = old_embeds.weight[: args.N_NEW_TOKENS, :].clone().detach()
    learnable_E = learnable_E[torch.randperm(args.N_NEW_TOKENS)]  # Random permutation
    learnable_E = torch.nn.Parameter(learnable_E)
    learnable_E.requires_grad = True
    # Replace input embeddings with extended version
    model.model.set_input_embeddings(
        ExtendedEmbedding(model.model.get_input_embeddings(), learnable_E)
    )

    # Create learnable output embeddings (unembeddings) similarly
    learnable_U = old_unembeds.weight[: args.N_NEW_TOKENS, :].clone().detach()
    learnable_U = learnable_U[torch.randperm(args.N_NEW_TOKENS)]  # Random permutation
    learnable_U = torch.nn.Parameter(learnable_U)
    learnable_U.requires_grad = True
    # Replace output embeddings with extended version
    model.model.set_output_embeddings(
        ExtendedLinear(model.model.get_output_embeddings(), learnable_U)
    )
    model.model.config.vocab_size = args.OFFSET + args.N_NEW_TOKENS
    model = model.to(device)

    # Setup optimizer for learnable embeddings only
    args.trainable_param_names = ".*new_weight.*"
    (optim, scheduler), trainable_param_names = get_optimizer_and_scheduler(
        model, args, -1
    )

    # Refresh LoRA layer references after model modifications
    lora_layers = get_lora_injected_layers(model)

    # Optional: Baseline evaluation (commented out for efficiency)
    """
    # how good is the model at the start?
    with set_active_expert(model, "fast_expert"):
        base_eval_loss = run_evaluation(model, dm.test_dataloader())
        logger.info(f"New expert evaluation loss: {base_eval_loss}")
    with set_active_expert(model, "oracle"):
        base_eval_loss = run_evaluation(model, dm.test_dataloader())
        logger.info(f"Oracle evaluation loss: {base_eval_loss}")
    with disable_modifiers(model):
        base_eval_loss = run_evaluation(model, dm.test_dataloader())
        logger.info(f"Base evaluation loss: {base_eval_loss}")
    """

    # Main training loop for dataset distillation
    for outer_it in range(args.total_steps):

        # Periodically evaluate the fast expert's adaptation capability
        if (outer_it + 1) % args.eval_every == 0:
            train_and_eval(model, dm.test_dataloader())
            lora_layers = get_lora_injected_layers(model)  # Refresh layer references

        # Inner loop: optimize embeddings based on gradient matching
        for inner_it in range(args.n_inner_steps):
            # Create synthetic batch using learnable embeddings
            batch = create_batch(args, labels=True)
            batch = transfer_batch_to_device(batch, model.device)

            # Forward pass through base model (no expert modifications)
            with disable_modifiers(model):
                outputs = model(**batch)

            # Compute gradients w.r.t. model weights
            outputs.loss.backward(retain_graph=True, create_graph=True)

            # Calculate loss based on gradient-LoRA matrix alignment
            cosim_loss, l1_loss, l2_loss = get_grad_loss(lora_layers)
            layer_losses = (
                cosim_loss  # Primary loss: cosine similarity between gradients and LoRA
            )
            
            # Log training metrics
            metric_logger.update(
                {
                    "cossim_loss": cosim_loss.item(),
                    "l1_loss": l1_loss.item(),
                    "l2_loss": l2_loss.item(),
                }
            )
            logger.info(
                f"Step {outer_it} Losses ({metric_logger.cossim_loss.avg:.4f}, {metric_logger.l1_loss.avg:.2f}, {metric_logger.l2_loss.avg:.2}), lr {optim.param_groups[0]['lr']}"
            )

            # Update learnable embeddings to improve gradient-LoRA alignment
            optim.zero_grad()
            model.zero_grad()
            layer_losses.backward()
            optim.step()
            scheduler.step()

            del outputs, layer_losses  # Free memory

    # Final evaluation with base model (no expert modifications)
    with disable_modifiers(model):
        base_eval_loss = run_evaluation(model, dm.test_dataloader())
        logger.info(f"Base evaluation loss: {base_eval_loss}")


if __name__ == "__main__":
    args = DSDistillConfig.parse()
    ds_distill(args)
