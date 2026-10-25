from modules.module.quantized.LinearGGUFA8 import LinearGGUFA8RequantFunction
from modules.module.quantized.LinearNf4 import LinearNf4

import torch
from torch import Tensor


class LinearNf4A8(LinearNf4):
    #Same strategy as LinearGGUFA8: the packed weight is dequantized to the compute dtype and
    #requantized axiswise to int8 per pass, so the int8 tensor-core path runs. The only difference
    #from LinearGGUFA8 is where the dense weight comes from: bitsandbytes NF4 dequantization
    #instead of GGUF block dequantization. The packed NF4 state (weight/absmax/code buffers) is
    #kept as-is, so offloading and re-quantization behave exactly like LinearNf4.
    #`dtype` selects the activation/weight 8-bit requantization dtype, like LinearGGUFA8:
    #int8 for NFLOAT_4_A8_INT, float8_e4m3fn for NFLOAT_4_A8_FLOAT
    def __init__(self, dtype: torch.dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert dtype in [torch.int8, torch.float8_e4m3fn]
        self._dtype = dtype

    def forward(self, x_orig: Tensor) -> Tensor:
        assert not self.weight.requires_grad
        x = x_orig.reshape(-1, x_orig.shape[-1])
        x = x.to(dtype=self.compute_dtype)

        w = self.unquantized_weight(self.compute_dtype, self.weight.device)
        #the 8-bit path only pays for itself above a few tokens, and an unquantized layer has
        #no packed NF4 weight to dequantize - the dense weight is used as-is
        if x.shape[0] > 16 and self.is_quantized:
            y = LinearGGUFA8RequantFunction.apply(x, w, self._dtype, self.bias, self.compute_dtype)
        else:
            y = torch.nn.functional.linear(x, w, self.bias)

        return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))
