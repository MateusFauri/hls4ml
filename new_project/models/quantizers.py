from qkeras import quantizers
from dataclasses import dataclass

@dataclass(frozen=True)
class Quantizers:
    weights: object
    activation: object


def build_quantizers(bits: int) -> Quantizers:

    if bits == 1:
        return Quantizers(
            weights=quantizers.binary(alpha=1),
            activation=quantizers.binary(alpha=1),
        )

    return Quantizers(
        weights=quantizers.quantized_bits(bits, 0, alpha=1),
        activation=quantizers.quantized_bits(bits, 0, alpha=1),
    )

def kernel_quantizer(bits: int):
    """Quantizer for weights."""

    return quantizers.quantized_bits(
        bits=bits,
        integer=0,
        alpha=1,
    )


def bias_quantizer(bits: int):
    """Quantizer for bias."""

    return quantizers.quantized_bits(
        bits=bits,
        integer=0,
        alpha=1,
    )


def activation_quantizer(bits: int):
    """Quantizer for activations."""

    return quantizers.quantized_relu(
        bits=bits,
        integer=0,
    )

