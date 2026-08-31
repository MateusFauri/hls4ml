from qkeras import quantizers
from dataclasses import dataclass

@dataclass(frozen=True)
class Quantizers:
    weights: object
    activation: object


def build_quantizers(bits: int) -> Quantizers:

    if bits == 1:
        q = quantizers.binary(alpha=1)
        return Quantizers(weights=q, activation=q)

    return Quantizers(
        weights=quantizers.quantized_bits(bits, bits - 1, alpha=1),
        activation=quantizers.quantized_bits(bits, bits - 1, alpha=1),
        #activation=quantizers.quantized_relu(bits),
    )
