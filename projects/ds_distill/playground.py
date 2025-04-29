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
    silence_logger,
    overfit_expert,
    soft_cross_entropy_loss, 
    entropy as entropy_fn,
    grad_alignment
)

@dataclass
class DSDistillConfig(ExpertConfig):
    tie_input_outputs: bool = False
    n_inner_steps: int = 1
    normalize_embeddings: bool = False
    prefix_length: int = 0
    n_samples: int = 10
    seq_len: int = 64
    # NOTE : left padding for now
    padding_side: str = "left"


def train_and_eval(model, eval_dataloader):
    fast_expert = model.get_expert_instance("new_expert")
    lora_layers = get_lora_injected_layers(model)
    reset_lora_params(fast_expert)

    fast_expert_params = [
        param for name, param in model.named_parameters() if "new_expert" in name
    ]
    for f_pam in fast_expert_params:
        f_pam.requires_grad = True
    optim = torch.optim.Adam(fast_expert_params, lr=5e-5)

    args.trainable_param_names = ".*new_expert.*"
    args.learning_rate = 5e-3
    args.total_steps = 5

    with silence_logger():
        (optim, scheduler), trainable_param_names = get_optimizer_and_scheduler(
            model, args, -1
        )

    with set_active_expert(model, "new_expert"):
        loss = run_evaluation(model, eval_dataloader)
        logger.info(f"\tEvaluation loss: {loss} before training")

        for it in range(args.total_steps):
            batch = create_batch(args, labels=False)
            batch = transfer_batch_to_device(batch, model.device)
            outputs = model(**batch)

            with set_active_expert(model, "oracle"):
                oracle_outputs = model(**batch)
                loss, entropy = soft_cross_entropy_loss(
                    oracle_outputs["logits"], outputs["logits"], batch['attention_mask']
                )

            optim.zero_grad()
            loss.backward()

            # check grad alignment
            grad_align = grad_alignment(lora_layers, oracle_expert_name="oracle")
            grad_align = grad_align['total'].detach()

            optim.step()
            scheduler.step()
            loss = run_evaluation(model, eval_dataloader)
            logger.info(
                f"\tInner Evaluation loss: {loss} at step {it} lr {optim.param_groups[0]['lr']}\t grad align {grad_align:.7f}"
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
    for arg in ["subsample_test", "predict_batch_size", "model", "padding_side"]:
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
        expert_name="overfit_one_sample", expert_config=expert.expert_config
    )
    model.add_empty_expert(
        expert_name="new_expert", expert_config=expert.expert_config
    )

    # set all parameters to not require gradients
    for name, param in model.named_parameters():
        param.requires_grad = False

    # add the hooks
    lora_layers = get_lora_injected_layers(model)

    # build the datamodule initially used to train the expert
    dm = get_datamodule(train_cfg)

    # Let's overfit the expert
    overfit_dl = overfit_expert(model, 'overfit_one_sample', train_cfg, args)
    overfit_batch = next(iter(overfit_dl))
    overfit_batch = transfer_batch_to_device(overfit_batch, model.device)

    # how good is the model at the start?
    with disable_modifiers(model):
        base_eval_loss = run_evaluation(model, dm.test_dataloader())
        logger.info(f"Base evaluation loss: {base_eval_loss}")
    with set_active_expert(model, "oracle"):
        oracle_eval_loss = run_evaluation(model, dm.test_dataloader())
        logger.info(f"Oracle evaluation loss: {oracle_eval_loss}")
        oracle_outputs = model(**{'input_ids': overfit_batch['input_ids'], 'attention_mask': overfit_batch['attention_mask']})
        oracle_ent = entropy_fn(oracle_outputs['logits'], overfit_batch['attention_mask']).item()
    with set_active_expert(model, "overfit_one_sample"):
        overfit_eval_loss = run_evaluation(model, dm.test_dataloader())
        overfit_train_loss = run_evaluation(model, overfit_dl)
        logger.info(f"Overfit evaluation loss: {overfit_eval_loss}, train loss: {overfit_train_loss}")
        
        overfit_outputs = model(**{'input_ids': overfit_batch['input_ids'], 'attention_mask': overfit_batch['attention_mask']})
        overfit_ent = entropy_fn(overfit_outputs['logits'], overfit_batch['attention_mask']).item()

    TARGET_EXPERT_NAME = 'oracle'
    TARGET_ENT = {'overfit_one_sample': overfit_ent, 'oracle': oracle_ent}[TARGET_EXPERT_NAME]

    # easy way to get the label indices
    args.N_NEW_TOKENS = args.n_samples * args.seq_len
    old_embeds = model.model.get_input_embeddings()
    old_unembeds = model.model.get_output_embeddings()
    args.OFFSET = old_embeds.num_embeddings
    logger.info(f"Added {args.N_NEW_TOKENS} new tokens to the model")

    # Build the learnable embeddings
    learnable_E = old_embeds.weight[: args.N_NEW_TOKENS, :].clone().detach()
    # shuffle on the first axis
    learnable_E = learnable_E[torch.randperm(args.N_NEW_TOKENS)]
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

    for outer_it in range(args.total_steps):

        if (outer_it + 1) % args.eval_every == 0:
            train_and_eval(model, dm.test_dataloader())
            lora_layers = get_lora_injected_layers(model)

        batch = create_batch(args, labels=False)
        batch = transfer_batch_to_device(batch, model.device)

        with set_active_expert(model, TARGET_EXPERT_NAME):
            oracle_outputs = model(**batch)
            entropy = entropy_fn(oracle_outputs["logits"], batch['attention_mask'])

        with disable_modifiers(model):
            base_outputs = model(**batch)
            base_entropy = entropy_fn(base_outputs["logits"], batch['attention_mask'])

        '''    
        with set_active_expert(model, "new_expert"):
            outputs = model(**batch)
            loss, entropy = soft_cross_entropy_loss(
                oracle_outputs["logits"], outputs["logits"], batch['attention_mask']
            )
            breakpoint()
            xx = 1
        '''


        loss = entropy.mean() - base_entropy.mean()
        metric_logger.update(
            {
                "entropy": entropy.mean().item(),
                "base_entropy": base_entropy.mean().item(),
                "entropy_delta": (entropy - base_entropy).mean().item(),
            }
        )
        logger.info(
            f"Step {outer_it} Losses ({metric_logger}), lr {optim.param_groups[0]['lr']}"
        )

        optim.zero_grad()
        model.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

    with disable_modifiers(model):
        base_eval_loss = run_evaluation(model, dm.test_dataloader())
        logger.info(f"Base evaluation loss: {base_eval_loss}")


if __name__ == "__main__":
    args = DSDistillConfig.parse()
    ds_distill(args)