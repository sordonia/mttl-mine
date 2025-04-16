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

N_NEW_TOKENS = None
OFFSET = None


@dataclass
class DSDistillConfig(ExpertConfig):
    tie_input_outputs: bool = False
    n_inner_steps: int = 1
    normalize_embeddings: bool = False
    prefix_length: int = 0
    n_samples: int = 10
    seq_len: int = 64


class ExtendedLinear(torch.nn.Module):
    def __init__(self, old_linear, new_weights):
        super().__init__()
        assert old_linear.bias is None
        self.weight = old_linear.weight
        self.new_weight = new_weights

    def forward(self, x):
        W = torch.cat([self.weight, self.new_weight], dim=0)
        return torch.nn.functional.linear(x, W)


class ExtendedEmbedding(torch.nn.Module):
    def __init__(self, old_embedding, new_weights):
        super().__init__()
        self.weight = old_embedding.weight
        self.new_weight = new_weights

    def forward(self, x):
        W = torch.cat([self.weight, self.new_weight], dim=0)
        return torch.nn.functional.embedding(x, W)


def get_lora_injected_layers(model):
    layers = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_a"):
            # TODO: double check this works for MultiExpertModel
            module.layer.weight.requires_grad = True
            module.layer.weight.retain_grad()
            layers[name] = module

    return layers


def get_grad_loss(lora_layers):
    sum_dotp = 0.0
    sum_ab_2 = 0.0
    sum_grad_2 = 0.0
    sum_l1 = 0
    sum_l2 = 0
    for name, lora_layer in lora_layers.items():
        if "layers.31" not in name:
            continue
        AB = (
            lora_layer.lora_b["oracle"].T @ lora_layer.lora_a["oracle"].T
        )  # (out_features, in_features)
        W_grad = lora_layer.weight.grad  # (out_features, in_features)
        sum_dotp += (AB * W_grad).sum()
        sum_ab_2 += (AB * AB).sum()
        sum_grad_2 += (W_grad * W_grad).sum()
        sum_l1 += (AB - W_grad).abs().sum()
        sum_l2 += (AB - W_grad).pow(2).sum()
    sum_l1 = sum_l1 / len(lora_layers)
    sum_l2 = sum_l2 / len(lora_layers)

    cos_sim = sum_dotp / (sum_ab_2 * sum_grad_2).sqrt()
    return 1 - cos_sim, sum_l1, sum_l2


def create_batch():
    global N_NEW_TOKENS, OFFSET
    # sample a batch of data
    idx = torch.randint(0, args.n_samples, (args.train_batch_size,))
    # idx = torch.arange(0, args.n_samples) #

    # expand idx into input_ids
    input_ids = idx.view(-1, 1) * args.seq_len
    offset = torch.arange(0, args.seq_len).reshape(1, -1)
    input_ids = input_ids + offset + OFFSET

    batch = {
        "input_ids": input_ids,
        "labels": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    return batch


@torch.no_grad()
def run_evaluation(model, dataloader):
    loss = 0
    pbar = tqdm(total=len(dataloader))
    for batch in dataloader:
        batch = transfer_batch_to_device(batch, model.device)
        with torch.no_grad():
            # make sure no task label is passed to the model
            outputs = model(
                **{"input_ids": batch["input_ids"], "labels": batch["labels"]}
            )
            loss += outputs.loss.item()
            pbar.update(1)
    pbar.close()
    return loss / len(dataloader)


def reset_lora_params(expert):
    # reset the weights of the fast expert
    # TODO: try and leverage the original weight initialization scheme for lora
    for p_name, param in expert.expert_weights.items():
        if "lora_a" in p_name:
            in_features, rank = param.size()
            assert in_features > rank
            gain = torch.nn.init.calculate_gain(
                nonlinearity="leaky_relu", param=math.sqrt(5)
            )
            std = gain / math.sqrt(in_features)
            with torch.no_grad():
                param.uniform_(-std, std)
        elif "lora_b" in p_name:
            param.zero_()


@contextmanager
def silence_logger():
    """
    Context manager to silence the logger.
    """
    import logging

    logger = logging.getLogger("mttl")
    old_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(old_level)


def train_and_eval(model, eval_dataloader):
    fast_expert = model.get_expert_instance("fast_expert")
    reset_lora_params(fast_expert)

    fast_expert_params = [
        param for name, param in model.named_parameters() if "fast_expert" in name
    ]
    for f_pam in fast_expert_params:
        f_pam.requires_grad = True
    optim = torch.optim.Adam(fast_expert_params, lr=5e-5)

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
            batch = create_batch()
            batch = transfer_batch_to_device(batch, model.device)
            outputs = model(**batch)
            optim.zero_grad()
            outputs.loss.backward()
            optim.step()
            scheduler.step()
            loss = run_evaluation(model, eval_dataloader)
            logger.info(
                f"\tInner Evaluation loss: {loss} at step {it} lr {optim.param_groups[0]['lr']}"
            )

        model.zero_grad()


def ds_distill(args: EvaluationConfig):
    global N_NEW_TOKENS, OFFSET

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
    N_NEW_TOKENS = args.n_samples * args.seq_len
    old_embeds = model.model.get_input_embeddings()
    old_unembeds = model.model.get_output_embeddings()
    OFFSET = old_embeds.num_embeddings

    # Build the learnable embeddings
    learnable_E = old_embeds.weight[:N_NEW_TOKENS, :].clone().detach()
    # shuffle on the first axis
    learnable_E = learnable_E[torch.randperm(N_NEW_TOKENS)]
    # learnable_E = learnable_E.reshape(args.n_samples, args.seq_len, -1)
    learnable_E = torch.nn.Parameter(learnable_E)
    learnable_E.requires_grad = True
    model.model.set_input_embeddings(
        ExtendedEmbedding(model.model.get_input_embeddings(), learnable_E)
    )

    # Build the learnable unembeddings
    learnable_U = old_unembeds.weight[:N_NEW_TOKENS, :].clone().detach()
    # shuffle on the first axis
    learnable_U = learnable_U[torch.randperm(N_NEW_TOKENS)]
    # learnable_U = learnable_U.reshape(args.n_samples, args.seq_len, -1)
    learnable_U = torch.nn.Parameter(learnable_U)
    learnable_U.requires_grad = True
    model.model.set_output_embeddings(
        ExtendedLinear(model.model.get_output_embeddings(), learnable_U)
    )
    model.model.config.vocab_size = OFFSET + N_NEW_TOKENS
    model = model.to(device)

    args.trainable_param_names = ".*new_weight.*"
    (optim, scheduler), trainable_param_names = get_optimizer_and_scheduler(
        model, args, -1
    )

    # Put this here so that Ws have require_grad = True
    lora_layers = get_lora_injected_layers(model)

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

    for outer_it in range(args.total_steps):

        if (outer_it + 1) % args.eval_every == 0:
            train_and_eval(model, dm.test_dataloader())
            lora_layers = get_lora_injected_layers(model)

        for inner_it in range(args.n_inner_steps):
            batch = create_batch()
            batch = transfer_batch_to_device(batch, model.device)

            with disable_modifiers(model):
                outputs = model(**batch)

            outputs.loss.backward(retain_graph=True, create_graph=True)

            cosim_loss, l1_loss, l2_loss = get_grad_loss(lora_layers)
            layer_losses = (
                cosim_loss  # l1_loss / l1_loss.item() # + l2_loss / l2_loss.item()
            )
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

            optim.zero_grad()
            model.zero_grad()
            layer_losses.backward()
            optim.step()
            scheduler.step()

            del outputs, layer_losses

    with disable_modifiers(model):
        base_eval_loss = run_evaluation(model, dm.test_dataloader())
        logger.info(f"Base evaluation loss: {base_eval_loss}")


if __name__ == "__main__":
    args = DSDistillConfig.parse()
    ds_distill(args)
