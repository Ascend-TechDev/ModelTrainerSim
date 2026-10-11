"""Number and byte formatting utilities."""

from __future__ import annotations


def format_bytes(num_bytes: float) -> str:
    """Human-readable byte string (B / KB / MB / GB / TB)."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} EB"


def format_number(n: float) -> str:
    """Human-readable count string (K / M / B / T)."""
    value = float(n)
    for suffix, divisor in (("", 1), ("K", 1e3), ("M", 1e6), ("B", 1e9), ("T", 1e12)):
        if abs(value) < divisor * 1000 or divisor == 1e12:
            if divisor == 1:
                return f"{value:,.0f}"
            return f"{value / divisor:.2f}{suffix}"
    return f"{value:.2f}T"


def format_flops(flops: float) -> str:
    """Human-readable FLOP/FLOPS string."""
    value = float(flops)
    for suffix, divisor in (("FLOPs", 1), ("KFLOPs", 1e3), ("MFLOPs", 1e6),
                            ("GFLOPs", 1e9), ("TFLOPs", 1e12), ("PFLOPs", 1e15)):
        if abs(value) < divisor * 1000 or divisor == 1e15:
            if divisor == 1:
                return f"{value:,.0f} {suffix}"
            return f"{value / divisor:.2f} {suffix}"
    return f"{value / 1e15:.2f} PFLOPs"