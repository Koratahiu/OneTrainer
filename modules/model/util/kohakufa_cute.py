"""Register the KohakuFA-CuTe attention backend into diffusers' dispatch registry.

KohakuFA-CuTe provides CuTe DSL attention kernels for Ampere/Ada/Hopper/Blackwell
GPUs with a stable large-logit backward. It is pulled in via the ``cuda`` extra
(``kohakufa-cute[cuda]``). When the package is unavailable (e.g. cpu/rocm envs)
registration is skipped and the backend is simply not offered.
"""

from typing import Any

import torch

from diffusers.models.attention_dispatch import (
    AttentionBackendName,
    _AttentionBackendRegistry,
    _check_device,
    _check_shape,
)

# The kohakufa_cute package is optional (CUDA-only). Guard the import so that
# non-CUDA environments can still import this module without the package.
try:
    from kohakufa_cute import automatic_scaled_dot_product_attention
except ImportError:
    automatic_scaled_dot_product_attention = None


def _register() -> bool:
    """Register the ``kohakufa_cute`` backend into the diffusers dispatch registry.

    Returns True if the backend was registered, False if the package is unavailable.
    """
    if automatic_scaled_dot_product_attention is None:
        return False

    # Add the 'kohakufa_cute' member to the diffusers AttentionBackendName enum.
    # `AttentionBackendName('kohakufa_cute')` raises ValueError for a member that
    # does not exist, so the member must be added explicitly to the enum's maps.
    if "kohakufa_cute" not in AttentionBackendName._member_map_:
        member = str.__new__(AttentionBackendName, "kohakufa_cute")
        member._value_ = "kohakufa_cute"
        member._name_ = "kohakufa_cute"
        AttentionBackendName._add_member_("kohakufa_cute", member)
        AttentionBackendName._value2member_map_["kohakufa_cute"] = member

    backend = AttentionBackendName("kohakufa_cute")

    # The function signature must match the kwargs that `dispatch_attention_fn`
    # passes, because diffusers filters kwargs against `inspect.signature(func)`.
    @ _AttentionBackendRegistry.register(
        backend,
        constraints=[_check_device, _check_shape],
    )
    def _kohakufa_cute(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        _parallel_config: Any | None = None,
    ) -> torch.Tensor:
        # diffusers dispatch passes BSHD [B, S, H, D]; the package expects BHSD [B, H, S, D].
        query = query.permute(0, 2, 1, 3).contiguous()
        key = key.permute(0, 2, 1, 3).contiguous()
        value = value.permute(0, 2, 1, 3).contiguous()

        # The package requires boolean masks; convert additive float masks.
        # Also reshape to a form broadcastable to [B, H, S_q, S_k].
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask == 0
            B, S_q, H, _ = query.shape
            S_k = key.shape[2]
            if attn_mask.ndim == 2 and attn_mask.shape == (B, S_k):
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            elif attn_mask.ndim == 3 and attn_mask.shape == (B, S_q, S_k):
                attn_mask = attn_mask.unsqueeze(1)

        out = automatic_scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        # Permute back to BSHD.
        return out.permute(0, 2, 1, 3)

    return True


# Register at import time so that importing this module makes the backend available.
_REGISTERED = _register()


def is_available() -> bool:
    """Whether the KohakuFA-CuTe backend is registered."""
    return _REGISTERED
