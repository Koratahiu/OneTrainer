import math
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.optim
import logging
import os
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any

# Add import for stochastic rounding
from modules.util.bf16_stochastic_rounding import add_stochastic_


class Prodigy8bit(torch.optim.Optimizer):
    r"""
    Implements Adam with Prodigy step-sizes.
    Leave LR set to 1 unless you encounter instability.
   
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the Prodigy learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        beta3 (float):
            coefficients for computing the Prodidy stepsize using running averages.
            If set to None, uses the value of square root of beta2 (default: None).
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        decouple (boolean):
            Use AdamW style decoupled weight decay
        use_bias_correction (boolean):
            Turn on Adam's bias correction. Off by default.
        safeguard_warmup (boolean):
            Remove lr from the denominator of D estimate to avoid issues during warm-up stage. Off by default.
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        d_coef (float):
            Coefficient in the expression for the estimate of d (default 1.0).
            Values such as 0.5 and 2.0 typically work as well. 
            Changing this parameter is the preferred way to tune the method.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
        slice_p (int): Reduce memory usage by calculating LR adaptation statistics on only every 
            pth entry of each tensor. For values greater than 1 this is an approximation to standard 
            Prodigy. Values ~11 are reasonable (default 1).
        stochastic_rounding (bool): Utilize stochastic rounding with BF16 on non-8bit params (default: False)
        min_8bit_size (int) The minimum size of a tensor before it is eligible to be quantized (default: 16384)
        quant_block_size (int) The amount of values to quantize into a single block (default: 2048)
        factored (bool): Use factored second moment estimation for memory savings (default: False)
        eps2 (float): Small constant added to factored second moment (default: 1e-30)
        clip_threshold (float): Clip updates to this value (default: None)
    """
    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), beta3=None,
                 eps=1e-8, weight_decay=0, decouple=True, 
                 use_bias_correction=False, safeguard_warmup=False,
                 d0=1e-6, d_coef=1.0, growth_rate=float('inf'),
                 fsdp_in_use=False,
                 slice_p=1,
                 stochastic_rounding=False,
                 min_8bit_size=16384,
                 quant_block_size=2048,
                 factored=False,
                 eps2=1e-30,
                 clip_threshold=1):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple and weight_decay > 0:
            print(f"Using decoupled weight decay")

        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        eps=eps, weight_decay=weight_decay,
                        d=d0, d0=d0, d_max=d0,
                        d_numerator=0.0, d_coef=d_coef,
                        k=0, growth_rate=growth_rate,
                        use_bias_correction=use_bias_correction,
                        decouple=decouple, safeguard_warmup=safeguard_warmup,
                        fsdp_in_use=fsdp_in_use,
                        slice_p=slice_p,
                        stochastic_rounding=stochastic_rounding,
                        min_8bit_size=min_8bit_size,
                        quant_block_size=quant_block_size,
                        factored=factored,
                        eps2=eps2,
                        clip_threshold=clip_threshold)
        self.d0 = d0
        self.stochastic_rounding = stochastic_rounding
        super().__init__(params, defaults)
        self.init_step()

    @property
    def supports_memory_efficient_fp16(self):
        # We will convert our data to/from float32
        # instead of having that done for us.
        return True

    @property
    def supports_flat_params(self):
        # We do not support using a single contiguous
        # Tensor for all of the parameters.
        return False

    @staticmethod
    def _should_use_matrix_factorization(grad_shape: torch.Size):
        grad_shape_dimensions = len(grad_shape)
        return grad_shape_dimensions == 2 or \
                (grad_shape_dimensions == 4 and grad_shape[2] == 1 and grad_shape[3] == 1)

    @staticmethod
    def _should_quantize_param(grad_shape: torch.Size, min_8bit_size: int):
        # We want to quantize blocks that have larger than `min_8bit_size`, but
        # only if they are `linear` or `1x1 convolution` layers.
        if Prodigy._should_use_matrix_factorization(grad_shape):
            return grad_shape.numel() > min_8bit_size
        return False

    @staticmethod
    def _quantize_param(params: torch.Tensor, quant_block_size: int):
        # Quantize our values in normalized `quant_block_size`-sized blocks.
        if params.numel() <= 1:
            return params

        data_chunk_list = params.split(quant_block_size)
        quantized_values: list = [None] * len(data_chunk_list)
        for index, data_chunk in enumerate(data_chunk_list, start=0):
            max_value = data_chunk.max()
            min_value = data_chunk.min()
            normalize_scale = (max_value - min_value) / 255.0

            values = ((data_chunk - min_value) / normalize_scale).round().byte()

            quantized_values[index] = {"value": values, "scale": normalize_scale, "min": min_value}

        return quantized_values

    @staticmethod
    def _dequantize_param(quantized_value_list):
        # If this isn't a quantized list, give it back
        if not isinstance(quantized_value_list, list):
            return quantized_value_list

        dequantized_values: list = [None] * len(quantized_value_list)
        for index, quantized_chunk in enumerate(quantized_value_list, start=0):
            dequantized_values[index] = \
                (quantized_chunk["value"].float() * quantized_chunk["scale"]) + quantized_chunk["min"]

        return torch.cat(dequantized_values)

    @staticmethod
    def _approx_sqrt(row, col):
        r_factor = (row / row.mean(dim=-1, keepdim=True)).sqrt_().unsqueeze(-1)
        c_factor = col.unsqueeze(-2).sqrt()
        return torch.mul(r_factor, c_factor)

    @staticmethod
    def _rms(tensor):
        return torch.linalg.norm(tensor) / (tensor.numel() ** 0.5)

    def init_step(self):
        self.d_denom = 0.0
        self.delta_numerator = 0.0

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        self.beta1, self.beta2 = group['betas']
        self.beta3 = group['beta3']
        if self.beta3 is None:
            self.beta3 = math.sqrt(self.beta2)
        k = group['k']

        self.d = group['d']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - self.beta2**(k+1))**0.5) / (1 - self.beta1**(k+1))
        else:
            bias_correction = 1

        self.dlr = self.d*lr*bias_correction
       
        self.decouple = group['decouple']
        self.fsdp_in_use = group['fsdp_in_use']
        self.factored = group['factored']
        self.eps2 = group['eps2']
        self.clip_threshold = group['clip_threshold']
        self.min_8bit_size = group['min_8bit_size']
        self.quant_block_size = group['quant_block_size']

        self.d_numerator = group['d_numerator']
        self.d_numerator *= self.beta3

        for group in self.param_groups:
            group_lr = group['lr']
            if group_lr not in [lr, 0.0]:
                raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")

    def step_parameter(self, p, group, i):
        if p.grad is None:
            return
        decay = group['weight_decay']
        k = group['k']
        eps = group['eps']
        group_lr = group['lr']
        d0 = group['d0']
        safeguard_warmup = group['safeguard_warmup']
        slice_p = group['slice_p']
        stochastic_rounding = group['stochastic_rounding']
        if hasattr(p, "_fsdp_flattened"):
            self.fsdp_in_use = True

        grad = p.grad.data

        # Apply weight decay (coupled variant)
        if decay != 0 and not self.decouple:
            if p.dtype == torch.bfloat16 and stochastic_rounding:
                add_stochastic_(grad, p.data, alpha=decay)
            else:
                grad.add_(p.data, alpha=decay)

        state = self.state[p]

        # Check if we should quantize this parameter
        should_quantize_param = Prodigy._should_quantize_param(grad.shape, group['min_8bit_size'])

        # State initialization
        if 'step' not in state:
            state['step'] = 0
            state['s'] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()
            if p.count_nonzero() > 0:
                state['p0'] = p.flatten()[::slice_p].detach().clone()
            else:
                state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)

            # Exponential moving average of gradient values
            if self.beta1 > 0:
                state['exp_avg'] = torch.zeros_like(p.data).detach() if not should_quantize_param else \
                                  Prodigy._quantize_param(torch.zeros_like(p.data), group['quant_block_size'])
            # Exponential moving average of squared gradient values
            if not self.factored or len(p.shape) < 2:
                state['exp_avg_sq'] = torch.zeros_like(p.data).detach() if not should_quantize_param else \
                                      Prodigy._quantize_param(torch.zeros_like(p.data), group['quant_block_size'])
            else:
                state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1]).to(grad)
                state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:]).to(grad)

        s = state['s']
        p0 = state['p0']

        if group_lr > 0.0:
            # we use d / d0 instead of just d to avoid getting values that are too small
            sliced_grad = grad.flatten()[::slice_p]
            self.delta_numerator += (self.d / d0) * self.dlr * torch.dot(sliced_grad, p0.data - p.data.flatten()[::slice_p]).item()

            # Adam EMA updates
            if self.beta1 > 0:
                exp_avg = state['exp_avg'] if not should_quantize_param else \
                          Prodigy._dequantize_param(state['exp_avg'])
                exp_avg.mul_(self.beta1).add_(grad, alpha=self.d * (1-self.beta1))
                state['exp_avg'] = exp_avg if not should_quantize_param else \
                                   Prodigy._quantize_param(exp_avg, group['quant_block_size'])

            if not self.factored or len(p.shape) < 2:
                exp_avg_sq = state['exp_avg_sq'] if not should_quantize_param else \
                             Prodigy._dequantize_param(state['exp_avg_sq'])
                exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=self.d * self.d * (1-self.beta2))
                state['exp_avg_sq'] = exp_avg_sq if not should_quantize_param else \
                                      Prodigy._quantize_param(exp_avg_sq, group['quant_block_size'])
            else:
                exp_avg_sq_row = state["exp_avg_sq_row"]
                exp_avg_sq_col = state["exp_avg_sq_col"]

                grad_sq = grad.square()
                exp_avg_sq_row.mul_(self.beta2).add_(grad_sq.mean(dim=-1), alpha=self.d * self.d * (1-self.beta2)).add_(self.eps2)
                exp_avg_sq_col.mul_(self.beta2).add_(grad_sq.mean(dim=-2), alpha=self.d * self.d * (1-self.beta2)).add_(self.eps2)

            if safeguard_warmup:
                s.mul_(self.beta3).add_(sliced_grad, alpha=((self.d / d0) * self.d))
            else:
                s.mul_(self.beta3).add_(sliced_grad, alpha=((self.d / d0) * self.dlr))
            self.d_denom += s.abs().sum().item()

        state['step'] += 1

        # Prepare denominator
        if not self.factored or len(p.shape) < 2:
            exp_avg_sq = state['exp_avg_sq'] if not should_quantize_param else \
                         Prodigy._dequantize_param(state['exp_avg_sq'])
            denom = exp_avg_sq.sqrt().add_(self.d * eps)
        else:
            denom = self._approx_sqrt(state["exp_avg_sq_row"], state["exp_avg_sq_col"]).add_(self.d * eps)

        # Apply weight decay (decoupled variant)
        if decay != 0 and self.decouple:
            if p.dtype == torch.bfloat16 and stochastic_rounding:
                add_stochastic_(p.data, p.data, alpha=-decay * self.dlr)
            else:
                p.data.add_(p.data, alpha=-decay * self.dlr)

        ### Take step
        if self.clip_threshold is None:
            if self.beta1 > 0:
                exp_avg = state['exp_avg'] if not should_quantize_param else \
                          Prodigy._dequantize_param(state['exp_avg'])
                if p.dtype == torch.bfloat16 and stochastic_rounding:
                    update = exp_avg / denom
                    add_stochastic_(p.data, -update, alpha=self.dlr)
                else:
                    p.data.addcdiv_(exp_avg, denom, value=-self.dlr)
            else:
                if p.dtype == torch.bfloat16 and stochastic_rounding:
                    update = grad / denom
                    add_stochastic_(p.data, -update, alpha=self.dlr * self.d)
                else:
                    p.data.addcdiv_(grad, denom, value=-self.dlr * self.d)
        else:
            if self.beta1 > 0:
                exp_avg = state['exp_avg'] if not should_quantize_param else \
                          Prodigy._dequantize_param(state['exp_avg'])
                update = exp_avg.div(denom)
            else:
                update = grad.div(denom).mul_(self.d)
            clip_div = (self._rms(update) / self.clip_threshold).clamp_(min=1.0)
            p.data.add_(update, alpha=-self.dlr / clip_div)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                self.step_parameter(p, group, i)

        self.calculate_d()
        self.init_step() #first init_step is called in __init__
        return loss
        
    def calculate_d(self):
        group = self.param_groups[0]
        d_max = group['d_max']
        d_coef = group['d_coef']
        growth_rate = group['growth_rate']
        lr = max(group['lr'] for group in self.param_groups)

        d_hat = self.d

        # if we have not done any progres, return
        # if we have any gradients available, will have d_denom > 0 (unless \|g\|=0)
        if self.d_denom == 0 and not self.fsdp_in_use:
            return

        if lr > 0.0:
            if self.fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = self.delta_numerator
                dist_tensor[1] = self.d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = self.d_numerator + dist_tensor[0]
                global_d_denom = dist_tensor[1]
            else:
                global_d_numerator = self.d_numerator + self.delta_numerator
                global_d_denom = self.d_denom

            d_hat = d_coef * global_d_numerator / global_d_denom
            if self.d == group['d0']:
                self.d = max(self.d, d_hat)
            d_max = max(d_max, d_hat)
            self.d = min(d_max, self.d * growth_rate)

        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d_denom'] = global_d_denom
            group['d'] = self.d
            group['d_max'] = d_max
            group['d_hat'] = d_hat
            group['k'] = group['k'] + 1

    def load_state_dict(self, state_dict):
        # Load the model's data
        super().load_state_dict(state_dict)

        # Reinitialize existing quantized values (lists of objects) as a byte
        # Reinitialize existing unquantized values (tensors) as a float
        quantizable_value_keys = [
            "exp_avg",
            "exp_avg_sq",
            "exp_avg_sq_col", "exp_avg_sq_row",
            "exp_avg_res_col", "exp_avg_res_row",
        ]
        for state in self.state.values():
            for quant_state_key in quantizable_value_keys:
                if quant_state_key in state:
                    if isinstance(state[quant_state_key], list):
                        for quantized_object in state[quant_state_key]:
                            quantized_object["value"] = quantized_object["value"].byte()
                    elif isinstance(state[quant_state_key], torch.Tensor):
                        state[quant_state_key] = state[quant_state_key].float()