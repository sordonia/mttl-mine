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
from projects.ds_distill.train_distill import (
    DSDistillConfig,
    ExtendedEmbedding,
    ExtendedLinear,
    get_lora_injected_layers,
    reset_lora_params,
    run_evaluation,
    silence_logger,
)


@torch.no_grad()
def run_evaluation_sce(model, dataloader):
    loss = 0
    pbar = tqdm(total=len(dataloader))
    for batch in dataloader:
        batch = transfer_batch_to_device(batch, model.device)
        with torch.no_grad():
            with set_active_expert(model, "oracle"):
                with torch.no_grad():
                    oracle_outputs = model(**batch)

            outputs = model(**batch)
            sce_loss, entropy = soft_cross_entropy_loss(
                oracle_outputs["logits"], outputs["logits"]
            )
            loss += sce_loss
            pbar.update(1)
    pbar.close()
    return loss / len(dataloader)


def create_batch(args):
    # sample a batch of data
    if args.n_samples == args.train_batch_size:
        idx = torch.arange(0, args.n_samples)
    else:
        idx = torch.randint(0, args.n_samples, (args.train_batch_size,))

    # expand idx into input_ids
    input_ids = idx.view(-1, 1) * args.seq_len
    offset = torch.arange(0, args.seq_len).reshape(1, -1)
    input_ids = input_ids + offset + args.OFFSET

    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    return batch


def soft_cross_entropy_loss(target_logits, trainable_logits, last_k_tokens=10):
    # Apply softmax to target_logits to get soft labels (probabilities)
    soft_targets = F.softmax(target_logits, dim=-1)

    # Apply log_softmax to trainable_logits
    log_probs = F.log_softmax(trainable_logits, dim=-1)

    # Compute cross-entropy loss
    # We use the formula: -sum(soft_targets * log_probs) averaged over batch and sequence
    if last_k_tokens > 0:
        # Only consider the last k tokens
        soft_targets = soft_targets[:, -last_k_tokens:, :]
        log_probs = log_probs[:, -last_k_tokens:, :]
    loss = -(soft_targets * log_probs).sum(dim=-1).mean()

    # We want the teacher (target logits) to be confident in its predictions
    # TODO: apply a entropy penalty to the soft targets
    entropy_loss = -torch.sum(soft_targets * torch.log(soft_targets + 1e-10), dim=-1)
    # loss -= entropy_loss.sum(1).mean()

    return loss, entropy_loss


def train_and_eval(model, eval_dataloader):
    fast_expert = model.get_expert_instance("fast_expert")
    reset_lora_params(fast_expert)

    fast_expert_params = [
        param for name, param in model.named_parameters() if "fast_expert" in name
    ]
    for f_pam in fast_expert_params:
        f_pam.requires_grad = True
    optim = torch.optim.Adam(fast_expert_params, lr=5e-6)

    args.trainable_param_names = ".*fast_expert.*"
    args.learning_rate = 5e-3
    args.total_steps = 5

    with silence_logger():
        (optim, scheduler), trainable_param_names = get_optimizer_and_scheduler(
            model, args, -1
        )

    with set_active_expert(model, "fast_expert"):
        loss = run_evaluation(model, eval_dataloader)
        logger.info(f"\tEvaluation loss: {loss} before training")

        for it in range(args.total_steps):
            batch = create_batch(args)
            batch = transfer_batch_to_device(batch, model.device)

            with set_active_expert(model, "oracle"):
                with torch.no_grad():
                    oracle_outputs = model(**batch)

            outputs = model(**batch)
            loss, entropy = soft_cross_entropy_loss(
                oracle_outputs["logits"], outputs["logits"]
            )

            # Now, compute cross entropy loss with soft labels
            # KL divergence

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            loss = run_evaluation(model, eval_dataloader)
            logger.info(
                f"\tInner Evaluation loss: {loss} at step {it} lr {optim.param_groups[0]['lr']}"
            )

        model.zero_grad()


def ds_distill(args: EvaluationConfig):
    seed_everything(args.seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))
    metric_logger = MetricLogger()

    remote_login(args.remote_token)

    assert (
        len(args.finetune_task_name.split(",")) == 1
    ), "Please provide a single expert selection for which to extract info"

    library = ExpertLibrary.get_expert_library(
        repo_id=args.library_id,
        token=args.remote_token,
        destination_id=args.destination_library_id,
        selection=args.finetune_task_name,
    )
    expert = library[args.finetune_task_name]
    train_cfg = ExpertConfig.from_dict(expert.training_config)

    # always overwrite these args
    for arg in ["subsample_test", "predict_batch_size", "model"]:
        if hasattr(args, arg) and getattr(args, arg) is not None:
            logger.info(f"Overriding {arg} with {getattr(args, arg)}")
            setattr(train_cfg, arg, getattr(args, arg))

    base_model = train_cfg.model

    loading_kwargs = {
        "device_map": args.device_map,
        "precision": "bf16",
        "attn_implementation": "flash_attention_2",
    }

    model = MultiExpertModel(
        MultiExpertModelConfig(
            base_model=base_model,
        ),
        **loading_kwargs,
    )

    model.add_expert_instance(expert, expert_name="oracle")
    model.add_empty_expert(
        expert_name="fast_expert", expert_config=expert.expert_config
    )

    # set all parameters to not require gradients
    for name, param in model.named_parameters():
        param.requires_grad = False

    # add the hooks
    lora_layers = get_lora_injected_layers(model)

    # build the datamodule initially used to train the expert
    dm = get_datamodule(train_cfg)

    # easy way to get the label indices
    args.N_NEW_TOKENS = args.n_samples * args.seq_len
    old_embeds = model.model.get_input_embeddings()
    old_unembeds = model.model.get_output_embeddings()
    args.OFFSET = old_embeds.num_embeddings

    # Build the learnable embeddings
    old_embed_norm = torch.norm(old_embeds.weight, dim=-1).mean()
    learnable_E = old_embeds.weight[: args.N_NEW_TOKENS, :].clone().detach()
    # shuffle on the first axis
    learnable_E = learnable_E[torch.randperm(args.N_NEW_TOKENS)]
    learnable_E = (
        torch.randn_like(learnable_E) * math.sqrt(1 / old_embeds.embedding_dim)
        + learnable_E
    )
    # learnable_E = learnable_E.reshape(args.n_samples, args.seq_len, -1)
    learnable_E = torch.nn.Parameter(learnable_E)
    learnable_E.requires_grad = True
    model.model.set_input_embeddings(
        ExtendedEmbedding(model.model.get_input_embeddings(), learnable_E)
    )

    # model.model.config.vocab_size = args.OFFSET + args.N_NEW_TOKENS
    model = model.to(device)

    args.trainable_param_names = ".*new_weight.*"
    (optim, scheduler), trainable_param_names = get_optimizer_and_scheduler(
        model, args, -1
    )

    # Put this here so that Ws have require_grad = True
    lora_layers = get_lora_injected_layers(model)

    # how good is the model at the start?
    """
    with set_active_expert(model, "fast_expert"):
        base_eval_loss = run_evaluation(model, dm.test_dataloader())
        logger.info(f"New expert evaluation loss: {base_eval_loss}")
    """
    with set_active_expert(model, "oracle"):
        base_eval_loss = run_evaluation(model, dm.test_dataloader())
        logger.info(f"Oracle evaluation loss: {base_eval_loss}")
        sce_eval_loss = run_evaluation_sce(model, dm.test_dataloader())
        logger.info(f"Oracle SCE evaluation loss: {sce_eval_loss}")
    """
    with disable_modifiers(model):
        base_eval_loss = run_evaluation(model, dm.test_dataloader())
        logger.info(f"Base evaluation loss: {base_eval_loss}")
    """

    for outer_it in range(args.total_steps):

        if (outer_it + 1) % args.eval_every == 0:
            train_and_eval(model, dm.test_dataloader())
            lora_layers = get_lora_injected_layers(model)

        for inner_it in range(args.n_inner_steps):
            batch = create_batch(args)
            batch = transfer_batch_to_device(batch, model.device)

            with disable_modifiers(model):
                outputs = model(**batch)

            with set_active_expert(model, "oracle"):
                oracle_outputs = model(**batch)

            KL = torch.nn.functional.kl_div(
                F.log_softmax(outputs["logits"], dim=-1),
                F.softmax(oracle_outputs["logits"], dim=-1),
                reduction="batchmean",
            )

            # we want to **maximize** the KL divergence
            loss = -KL
            # loss, entropy = soft_cross_entropy_loss(oracle_outputs["logits"], outputs["logits"])
            # loss =

            logger.info(
                f"Step {outer_it} Losses ({-1 * loss.item():.4f}) lr {optim.param_groups[0]['lr']}, learnable_E norm: {torch.norm(learnable_E.data, dim=-1).mean()} vs {old_embed_norm}"
            )

            optim.zero_grad()
            model.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            # reset the norm of learnable_E to
            # learnable_E.data.div_(torch.norm(learnable_E.data, dim=-1, keepdim=True)).mul_(old_embed_norm)

            del outputs, loss

    with disable_modifiers(model):
        base_eval_loss = run_evaluation(model, dm.test_dataloader())
        logger.info(f"Base evaluation loss: {base_eval_loss}")


if __name__ == "__main__":
    args = DSDistillConfig.parse()
    ds_distill(args)
