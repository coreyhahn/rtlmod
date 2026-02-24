"""Numpy-backed arbitrary-width integer arrays."""

from __future__ import annotations

import numpy as np


def _backing_dtype(width: int, signed: bool):
    """Choose the smallest numpy dtype that fits the given width."""
    if signed:
        if width <= 8:
            return np.int8
        if width <= 16:
            return np.int16
        if width <= 32:
            return np.int32
        return np.int64
    else:
        if width <= 8:
            return np.uint8
        if width <= 16:
            return np.uint16
        if width <= 32:
            return np.uint32
        return np.uint64


class RtlArray:
    """A numpy-backed array with arbitrary bit-width and Verilog masking semantics.

    Stores data in numpy arrays using int64 internally for arithmetic safety.
    After every operation, results are masked to enforce bit-width constraints.
    Single-element indexing returns typed scalars (UInt[N] or SInt[N]).
    Slice indexing returns a new RtlArray.
    """

    __slots__ = ("_data", "_width", "_signed", "_mask")

    def __init__(self, data: np.ndarray, width: int, signed: bool):
        self._data = data
        self._width = width
        self._signed = signed
        self._mask = (1 << width) - 1

    @property
    def shape(self):
        return self._data.shape

    @property
    def width(self):
        return self._width

    @property
    def signed(self):
        return self._signed

    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(result, np.ndarray):
            return RtlArray(result.copy(), self._width, self._signed)
        # Scalar element - return a typed scalar
        from rtlmod.types import SInt, UInt

        if self._signed:
            return SInt[self._width](int(result))
        return UInt[self._width](int(result))

    def __eq__(self, other):
        if isinstance(other, RtlArray):
            return (
                np.array_equal(self._data, other._data)
                and self._width == other._width
                and self._signed == other._signed
            )
        return NotImplemented

    def __repr__(self) -> str:
        kind = "SInt" if self._signed else "UInt"
        return f"RtlArray({kind}[{self._width}], shape={self.shape})"

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the underlying numpy data."""
        return self._data.copy()

    # --- Arithmetic ---

    @staticmethod
    def _apply_mask(data: np.ndarray, width: int, signed: bool) -> RtlArray:
        """Apply bit-width masking to numpy data, returning a new RtlArray."""
        mask = (1 << width) - 1
        arr = data.astype(np.int64)
        masked = arr & mask
        if signed:
            sign_bit = 1 << (width - 1)
            masked = np.where(masked >= sign_bit, masked - (1 << width), masked)
        return RtlArray(masked, width, signed)

    def _get_other_data(self, other):
        """Extract numpy data and type info from another RtlArray or scalar."""
        if isinstance(other, RtlArray):
            return other._data.astype(np.int64), other._width, other._signed
        from rtlmod.types import _SIntScalar, _UIntScalar

        if isinstance(other, (_UIntScalar, _SIntScalar)):
            return np.int64(other._to_int()), other._width, other.signed
        return NotImplemented, None, None

    def __add__(self, other):
        o_data, o_width, o_signed = self._get_other_data(other)
        if o_data is NotImplemented:
            return NotImplemented
        s = self._signed or o_signed
        w = max(self._width, o_width) + 1
        return self._apply_mask(self._data.astype(np.int64) + o_data, w, s)

    def __radd__(self, other):
        o_data, o_width, o_signed = self._get_other_data(other)
        if o_data is NotImplemented:
            return NotImplemented
        s = self._signed or o_signed
        w = max(self._width, o_width) + 1
        return self._apply_mask(o_data + self._data.astype(np.int64), w, s)

    def __sub__(self, other):
        o_data, o_width, o_signed = self._get_other_data(other)
        if o_data is NotImplemented:
            return NotImplemented
        w = max(self._width, o_width) + 1
        return self._apply_mask(self._data.astype(np.int64) - o_data, w, True)

    def __rsub__(self, other):
        o_data, o_width, o_signed = self._get_other_data(other)
        if o_data is NotImplemented:
            return NotImplemented
        w = max(self._width, o_width) + 1
        return self._apply_mask(o_data - self._data.astype(np.int64), w, True)

    def __mul__(self, other):
        o_data, o_width, o_signed = self._get_other_data(other)
        if o_data is NotImplemented:
            return NotImplemented
        s = self._signed or o_signed
        w = self._width + o_width
        return self._apply_mask(self._data.astype(np.int64) * o_data, w, s)

    def __rmul__(self, other):
        o_data, o_width, o_signed = self._get_other_data(other)
        if o_data is NotImplemented:
            return NotImplemented
        s = self._signed or o_signed
        w = self._width + o_width
        return self._apply_mask(o_data * self._data.astype(np.int64), w, s)

    def __lshift__(self, n):
        return self._apply_mask(
            self._data.astype(np.int64) << int(n), self._width, self._signed
        )

    def __rshift__(self, n):
        return self._apply_mask(
            self._data.astype(np.int64) >> int(n), self._width, self._signed
        )

    def __and__(self, other):
        o_data, o_width, o_signed = self._get_other_data(other)
        if o_data is NotImplemented:
            return NotImplemented
        w = max(self._width, o_width)
        s = self._signed and o_signed
        return self._apply_mask(self._data.astype(np.int64) & o_data, w, s)

    def __or__(self, other):
        o_data, o_width, o_signed = self._get_other_data(other)
        if o_data is NotImplemented:
            return NotImplemented
        w = max(self._width, o_width)
        s = self._signed and o_signed
        return self._apply_mask(self._data.astype(np.int64) | o_data, w, s)

    def __xor__(self, other):
        o_data, o_width, o_signed = self._get_other_data(other)
        if o_data is NotImplemented:
            return NotImplemented
        w = max(self._width, o_width)
        s = self._signed and o_signed
        return self._apply_mask(self._data.astype(np.int64) ^ o_data, w, s)
