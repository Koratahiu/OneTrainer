from collections.abc import Iterable

import torch


def is_geoembed_ema_active(optimizer) -> bool:
    """
    Returns True when the optimizer is a GeoEmbed_adv instance whose per-token
    geodesic EMA (``ema_beta``) is enabled (a value in [0, 1) that is > 0).
    """
    if optimizer is None:
        return False
    try:
        from adv_optm import GeoEmbed_adv
    except ImportError:
        return False
    if not isinstance(optimizer, GeoEmbed_adv):
        return False
    for group in optimizer.param_groups:
        beta = group.get("ema_beta")
        if beta is not None and beta > 0.0:
            return True
    return False


class GeoEMAModuleWrapper:
    """
    Drop-in replacement for ``EMAModuleWrapper`` that exposes the standard EMA
    interface on top of the ``GeoEmbed_adv`` optimizer's *internal* per-token
    geodesic EMA buffer (``state[p]["p_ema"]``).

    When ``GeoEmbed_adv`` is used with ``ema_beta`` enabled, this buffer *is* the
    EMA model: it is maintained inside the optimizer's own ``step()`` (bias-corrected
    SLERP on directions + log-Euclidean averaging on radii), so no separate
    ``EMAModuleWrapper`` copy is created. The trainer, sampler and saver keep using
    the usual ``step`` / ``copy_ema_to`` / ``copy_temp_to`` / ``get_current_decay``
    calls, which here delegate to the optimizer state. The EMA is persisted through
    the optimizer state (``p_ema``), not through this object's ``state_dict``.
    """

    def __init__(self, optimizer, parameters: Iterable[torch.nn.Parameter], device=None):
        self.optimizer = optimizer
        self.parameters = list(parameters)
        self.temp_stored_parameters = None
        self.device = device
        self.decay = self._read_ema_beta()

    def _read_ema_beta(self):
        for group in self.optimizer.param_groups:
            beta = group.get("ema_beta")
            if beta is not None:
                return beta
        return None

    def get_current_decay(self, optimization_step) -> float:
        return self.decay if self.decay is not None else 0.0

    @torch.no_grad()
    def step(self, parameters, optimization_step):
        # The GeoEmbed_adv optimizer updates its internal ``p_ema`` buffer inside
        # its own ``step()``; nothing extra is required here.
        return None

    def to(self, device=None, dtype=None) -> None:
        # The EMA buffer lives inside the optimizer state; there is no separate
        # copy to move.
        if device is not None:
            self.device = device

    @torch.no_grad()
    def copy_ema_to(self, parameters, store_temp: bool = True) -> None:
        parameters = list(parameters)
        if store_temp:
            self.temp_stored_parameters = [parameter.detach().cpu() for parameter in parameters]

        for parameter in parameters:
            state = self.optimizer.state.get(parameter)
            if state is not None and "p_ema" in state:
                parameter.data.copy_(state["p_ema"].to(parameter.device, dtype=parameter.dtype).data)

    @torch.no_grad()
    def copy_temp_to(self, parameters) -> None:
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters, strict=True):
            parameter.data.copy_(temp_parameter.to(parameter.device))

        self.temp_stored_parameters = None

    def load_state_dict(self, state_dict: dict) -> None:
        # The EMA buffer is restored through the optimizer state, not here.
        return None

    def state_dict(self) -> dict:
        # The EMA buffer is part of the optimizer state; nothing separate to save.
        return {}
