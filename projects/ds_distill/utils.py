import os
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
from contextlib import contextmanager

from mttl.logging import logger
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.utils import transfer_batch_to_device

from mttl.models.expert_model import (
    set_active_expert,
)
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



def train_expert(model, expert_name, dataloader, args, **kwargs):
    """
    Ability to train a single expert super quickly, to
    see if we can recover the updates made on a single expert
    """

    args = deepcopy(args)
    model.cuda()

    for key, value in kwargs.items():
        assert hasattr(args, key), f"args does not have {key}"
        logger.info(f'Setting {key} to {value}')
        setattr(args, key, value)

    args.trainable_param_names = f".*{expert_name}.*"
    # Build the expert model from the expert 

    with silence_logger():
        (optim, scheduler), trainable_param_names = get_optimizer_and_scheduler(
            model, args, -1
        )

    total_loss = 0.0
    progress_bar = tqdm(total=args.total_steps, desc="Training Progress", leave=True)
    iterator = iter(dataloader)

    for batch_idx in range(args.total_steps):
        if batch_idx >= args.total_steps:
            break

        try: 
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        batch = transfer_batch_to_device(batch, model.device)
        with set_active_expert(model, expert_name):
            outputs = model(**{'input_ids': batch['input_ids'], 'labels': batch['labels'], 'attention_mask': batch['attention_mask']})
            # outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

        progress_bar.set_postfix({"Loss": loss.item(), "Avg Loss": avg_loss})
        progress_bar.update(1)

    progress_bar.close()

def overfit_expert(model, expert_name, ds_cfg, args):
    from mttl.datamodule.base import get_datamodule

    ds_cfg = deepcopy(ds_cfg)
    ds_cfg.subsample_train = 1 
    dm = get_datamodule(ds_cfg)

    train_expert(
        model,
        expert_name,
        dm.train_dataloader(),
        args,
        total_steps=25,
    )

    return dm.train_dataloader() 

def reset_lora_params(expert):
    """
    Reset the LoRA parameters of the expert model.
    """
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


@torch.no_grad()
def run_evaluation(model, dataloader):

    """
    Run evaluation on the model with the task-specific dataloader
    NOTE: Assumes default expert has been set, as not task labels are provided
    """

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

def grad_alignment(lora_layers, oracle_expert_name="oracle"):
    """
    Measure the cosine similarity between the gradients & the LoRA parameters
    """
    sum_dotp = 0.0
    sum_ab_2 = 0.0
    sum_grad_2 = 0.0
    
    cossims = {}
    for name, lora_layer in lora_layers.items():
        AB = (
            lora_layer.lora_b[oracle_expert_name].T @ lora_layer.lora_a[oracle_expert_name].T
        )  # (out_features, in_features)
        W_grad = lora_layer.weight.grad  # (out_features, in_features)
        sum_dotp += (AB * W_grad).sum()
        sum_ab_2 += (AB * AB).sum()
        sum_grad_2 += (W_grad * W_grad).sum()
        cossims[name] = ((AB * W_grad).sum() / (
            (AB * AB).sum() * (W_grad * W_grad).sum()
        ).sqrt()) # .item()

    cossims['total']= (sum_dotp / (sum_ab_2 * sum_grad_2).sqrt()) # .item()
    return cossims

def create_batch(args, labels=False):
    # sample a batch of data
    if args.n_samples <= args.train_batch_size:
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

    if labels:
        batch['labels'] = input_ids.clone()

    return batch

def get_lora_injected_layers(model):
    layers = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_a"):
            # TODO: double check this works for MultiExpertModel
            module.layer.weight.requires_grad = True
            module.layer.weight.retain_grad()
            layers[name] = module

    return layers

def soft_cross_entropy_loss(target_logits, trainable_logits, attn_mask, last_k_tokens=-1):
    # Apply softmax to target_logits to get soft labels (probabilities)
    soft_targets = F.softmax(target_logits, dim=-1)

    # Apply log_softmax to trainable_logits
    log_probs = F.log_softmax(trainable_logits, dim=-1)

    # make sure we are left padded
    assert attn_mask[:, -1].all()

    # Compute cross-entropy loss
    # We use the formula: -sum(soft_targets * log_probs) averaged over batch and sequence
    if last_k_tokens > 0:
        attn_mask[:, :-last_k_tokens] = 0
    
    soft_targets = soft_targets[attn_mask != 0]
    log_probs = log_probs[attn_mask != 0]
    seq_lens = attn_mask.sum(dim=-1)

    # we want to sum over the sequence length, and average over the batch
    seq_deno = seq_lens.view(-1, 1).expand_as(attn_mask)[attn_mask != 0]
    
    loss = -(soft_targets * log_probs) / seq_deno.view(-1, 1)
    loss = loss.sum(dim=-1).mean()

    # We want the teacher (target logits) to be confident in its predictions
    # TODO: apply a entropy penalty to the soft targets
    entropy_loss = -torch.sum(soft_targets * torch.log(soft_targets + 1e-10), dim=-1)
    # loss -= entropy_loss.sum(1).mean()

    return loss, entropy_loss

def entropy(logits, attn_mask): 
    # Apply softmax to target_logits to get soft labels (probabilities)
    probs = F.softmax(logits, dim=-1)

    # make sure we are left padded
    # assert attn_mask[:, -1].all()

    # Compute cross-entropy loss
    # We use the formula: -sum(soft_targets * log_probs) averaged over batch and sequence
    attn_mask[:, :-1] = 0
    probs = probs[attn_mask != 0]
    seq_lens = attn_mask.sum(dim=-1)

    # we want to sum over the sequence length, and average over the batch
    seq_deno = seq_lens.view(-1, 1).expand_as(attn_mask)[attn_mask != 0]
    
    loss = -(probs * torch.log(probs + 1e-10)) / seq_deno.view(-1, 1)
    loss = loss.sum(dim=-1).mean()

    return loss

class ExtendedLinear(torch.nn.Module):
    def __init__(self, old_linear, new_weights):
        super().__init__()
        if old_linear.bias is not None:
            bias = torch.cat(
                [old_linear.bias, torch.zeros(new_weights.size(0), dtype=old_linear.bias.dtype)], dim=0
            )
            self.register_buffer("bias", bias)
        else:
            self.bias = None
           
        self.weight = old_linear.weight
        self.new_weight = new_weights

    def forward(self, x):
        W = torch.cat([self.weight, self.new_weight], dim=0)
        return torch.nn.functional.linear(x, W, bias=self.bias)


class ExtendedEmbedding(torch.nn.Module):
    def __init__(self, old_embedding, new_weights):
        super().__init__()
        self.weight = old_embedding.weight
        self.new_weight = new_weights

    def forward(self, x):
        W = torch.cat([self.weight, self.new_weight], dim=0)
        return torch.nn.functional.embedding(x, W)
