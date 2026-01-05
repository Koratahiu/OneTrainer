import math
from collections.abc import Callable
from typing import Any


def lr_lambda_warmup(
    warmup_steps: int, 
    lr_lambda: Callable[[int], float], 
    optimizer: Any = None
):
    # Capture initial betas/momentum from the optimizer at the time of creation
    initial_betas = []
    if False and optimizer is not None:
        for group in optimizer.param_groups:
            if "betas" in group:
                initial_betas.append(group["betas"])
            elif "momentum" in group:
                initial_betas.append(group["momentum"])
            elif "beta1" in group:
                initial_betas.append(group["beta1"])
            else:
                initial_betas.append(None)

    def warmup(current_step: int):
        if current_step < warmup_steps:
            progress = float(current_step) / float(warmup_steps)

            # Linearly warmup beta1/momentum if optimizer is provided
            if False and optimizer is not None:
                start_beta = 0.01 # Start near 0
                for i, group in enumerate(optimizer.param_groups):
                    target = initial_betas[i]
                    if target is None:
                        continue

                    if "betas" in group:
                        # target is (beta1, beta2, ...)
                        target_beta1 = target[0]
                        new_beta1 = start_beta + (target_beta1 - start_beta) * progress
                        # Reconstruct tuple with new beta1 and original beta2+
                        group["betas"] = (new_beta1,) + target[1:]
 
                    elif "momentum" in group:
                        target_momentum = target
                        new_momentum = start_beta + (target_momentum - start_beta) * progress
                        group["momentum"] = new_momentum

                    elif "beta1" in group:
                        target_beta1 = target
                        new_beta1 = start_beta + (target_beta1 - start_beta) * progress
                        group["beta1"] = new_beta1

            return progress
        else:
            # Ensure betas are restored/maintained at target once warmup is done
            if False and optimizer is not None:
                for i, group in enumerate(optimizer.param_groups):
                    target = initial_betas[i]
                    if target is None: 
                        continue

                    if "betas" in group:
                        if group["betas"] != target:
                            group["betas"] = target
                    elif "momentum" in group:
                        if group["momentum"] != target:
                            group["momentum"] = target
                    elif "beta1" in group:
                        if group["beta1"] != target:
                            group["beta1"] = target

            return lr_lambda(current_step - warmup_steps)

    return warmup


def lr_lambda_constant():
    def lr_lambda(current_step: int):
        return 1

    return lr_lambda


def lr_lambda_linear(
        scheduler_steps: int,
        min_factor: float = 1.0,
):
    def lr_lambda(current_step: int):
        lin_val = max(0.0, float(scheduler_steps - current_step) / float(scheduler_steps))
        factor = apply_min_factor(lin_val, min_factor)
        return factor

    return lr_lambda



def lr_lambda_cosine(
        scheduler_steps: int,
        min_factor: float = 1.0,
):
    def lr_lambda(current_step: int):
        progress = float(current_step) / float(scheduler_steps)
        cos_val = 0.5 * (1.0 + math.cos(progress * math.pi))
        factor = max(0.0, cos_val)
        factor = apply_min_factor(factor, min_factor)
        return factor

    return lr_lambda


def lr_lambda_cosine_with_restarts(
        scheduler_steps: int,
        num_cycles: float,
        min_factor: float = 1.0,
):
    def lr_lambda(current_step: int):
        progress = float(min(current_step, scheduler_steps - 1)) / float(scheduler_steps)
        cos_val = 0.5 * (1.0 + math.cos(progress * 2.0 * math.pi * num_cycles))
        factor = max(0.0, cos_val)
        factor = apply_min_factor(factor, min_factor)
        return factor

    return lr_lambda



def lr_lambda_cosine_with_hard_restarts(
        scheduler_steps: int,
        num_cycles: float,
        min_factor: float = 1.0,
):
    def lr_lambda(current_step: int):
        progress = float(min(current_step, scheduler_steps - 1)) / float(scheduler_steps)
        cos_val = 0.5 * (1.0 + math.cos(((progress * num_cycles) % 1.0) * math.pi))
        factor = max(0.0, cos_val)
        factor = apply_min_factor(factor, min_factor)
        return factor

    return lr_lambda



def lr_lambda_rex(
        scheduler_steps: int,
        min_factor: float = 1.0,
):
    def lr_lambda(current_step: int):
        # https://arxiv.org/abs/2107.04197
        max_lr = 1
        min_lr = 0
        d = 0.9
        if current_step < scheduler_steps:
            progress = (current_step / scheduler_steps)
            div = (1 - d) + (d * (1 - progress))
            val = min_lr + (max_lr - min_lr) * ((1 - progress) / div)
        else:
            val = min_lr

        factor = apply_min_factor(val, min_factor)
        return factor

    return lr_lambda

def lr_lambda_mdlrc(
        scheduler_steps: int,
        optimizer: Any,
        min_factor: float = 1.0,
        beta0: float = 0.9,
):
    def lr_lambda(current_step: int):
        # https://arxiv.org/abs/2404.07946v1
        tau = min(1.0, max(0.0, float(current_step) / float(scheduler_steps))) if scheduler_steps > 0 else 1.0
        # denom = (1.0 - beta0) + beta0 * (1.0 - tau)
        # beta_t = beta0 * (1.0 - tau) / denom
        # lr_scale = (1.0 - beta0) / (1.0 - beta_t)
        # Modified to be linear for both beta and lr
        beta_t = beta0 * (1.0 - tau)
        lr_scale = 1.0 - tau

        for group in optimizer.param_groups:
            if "betas" in group:
                betas = group["betas"]
                if len(betas) == 2:
                    group["betas"] = (beta_t, betas[1])
                elif len(betas) == 3:
                    group["betas"] = (beta_t, betas[1], betas[2])
            elif "momentum" in group:
                group["momentum"] = beta_t
            elif "beta1" in group:
                group["beta1"] = beta_t

        factor = apply_min_factor(lr_scale, min_factor)
        return factor

    return lr_lambda

def apply_min_factor(value: float, min_factor: float) -> float:
    return min_factor + (1.0 - min_factor) * value
