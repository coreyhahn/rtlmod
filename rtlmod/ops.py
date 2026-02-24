"""Bit manipulation operations."""

from __future__ import annotations


def concat(*args):
    """Concatenate values. First argument is MSB. Result is always unsigned."""
    from rtlmod.types import UInt
    total_width = sum(a.width for a in args)
    result = 0
    for a in args:
        result = (result << a.width) | a._to_unsigned()
    return UInt[total_width](result)
