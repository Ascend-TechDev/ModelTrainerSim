"""Data type definitions and helper utilities."""

from dataclasses import dataclass
from enum import Enum


class DType(str, Enum):
    """Supported data types for parameter and activation storage."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    FP4 = "fp4"
    FP4_HALF = "fp4_half"
    INT8 = "int8"
    INT4 = "int4"
    BOOL = "bool"


@dataclass(frozen=True)
class _DTypeInfo:
    bits: int
    bytes_per_element: float


_DTYPE_INFO: dict[DType, _DTypeInfo] = {
    DType.FP32: _DTypeInfo(bits=32, bytes_per_element=4.0),
    DType.FP16: _DTypeInfo(bits=16, bytes_per_element=2.0),
    DType.BF16: _DTypeInfo(bits=16, bytes_per_element=2.0),
    DType.FP8: _DTypeInfo(bits=8, bytes_per_element=1.0),
    DType.FP4: _DTypeInfo(bits=4, bytes_per_element=1.0),
    DType.FP4_HALF: _DTypeInfo(bits=4, bytes_per_element=0.5),
    DType.INT8: _DTypeInfo(bits=8, bytes_per_element=1.0),
    DType.INT4: _DTypeInfo(bits=4, bytes_per_element=0.5),
    DType.BOOL: _DTypeInfo(bits=1, bytes_per_element=0.125),
}


def get_dtype_size(dtype: DType | str) -> float:
    """Return the size in bytes for a single element of the given dtype."""
    if isinstance(dtype, str):
        dtype = DType(dtype)
    return _DTYPE_INFO[dtype].bytes_per_element


def get_dtype_bits(dtype: DType | str) -> int:
    """Return the size in bits for a single element of the given dtype."""
    if isinstance(dtype, str):
        dtype = DType(dtype)
    return _DTYPE_INFO[dtype].bits