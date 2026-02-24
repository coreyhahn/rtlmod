"""Arbitrary-width integer types for RTL modeling."""

from __future__ import annotations


class _ArithmeticMixin:
    """Shared arithmetic operators for UInt and SInt scalars.

    Implements Verilog-style width rules:
    - add/sub: max(wa, wb) + 1
    - mul: wa + wb
    - subtraction always produces signed result
    - if either operand is signed, result is signed (except sub, always signed)
    - bitwise: max(wa, wb), signed only if both signed
    - shifts: same width as left operand
    """

    def __add__(self, other):
        if not _is_rtl_type(other):
            return NotImplemented
        s = self.signed or other.signed
        w = max(self._width, other._width) + 1
        return _make_result(self._to_int() + other._to_int(), w, s)

    def __radd__(self, other):
        if not _is_rtl_type(other):
            return NotImplemented
        return other.__add__(self)

    def __sub__(self, other):
        if not _is_rtl_type(other):
            return NotImplemented
        w = max(self._width, other._width) + 1
        return _make_result(self._to_int() - other._to_int(), w, True)

    def __rsub__(self, other):
        if not _is_rtl_type(other):
            return NotImplemented
        return other.__sub__(self)

    def __mul__(self, other):
        if not _is_rtl_type(other):
            return NotImplemented
        s = self.signed or other.signed
        w = self._width + other._width
        return _make_result(self._to_int() * other._to_int(), w, s)

    def __rmul__(self, other):
        if not _is_rtl_type(other):
            return NotImplemented
        return other.__mul__(self)

    def __lshift__(self, n):
        return type(self)(self._to_int() << int(n))

    def __rshift__(self, n):
        return type(self)(self._to_int() >> int(n))

    def __and__(self, other):
        if not _is_rtl_type(other):
            return NotImplemented
        w = max(self._width, other._width)
        s = self.signed and other.signed
        return _make_result(self._to_unsigned() & other._to_unsigned(), w, s)

    def __rand__(self, other):
        if not _is_rtl_type(other):
            return NotImplemented
        return other.__and__(self)

    def __or__(self, other):
        if not _is_rtl_type(other):
            return NotImplemented
        w = max(self._width, other._width)
        s = self.signed and other.signed
        return _make_result(self._to_unsigned() | other._to_unsigned(), w, s)

    def __ror__(self, other):
        if not _is_rtl_type(other):
            return NotImplemented
        return other.__or__(self)

    def __xor__(self, other):
        if not _is_rtl_type(other):
            return NotImplemented
        w = max(self._width, other._width)
        s = self.signed and other.signed
        return _make_result(self._to_unsigned() ^ other._to_unsigned(), w, s)

    def __rxor__(self, other):
        if not _is_rtl_type(other):
            return NotImplemented
        return other.__xor__(self)

    def __invert__(self):
        return type(self)(~self._to_unsigned())

    def __neg__(self):
        if self.signed:
            return type(self)(-self._value)
        return SInt[self._width](-self._to_int())


class _IntTypeMeta(type):
    """Metaclass enabling UInt[N] subscript syntax with type caching."""

    _cache: dict[tuple[str, int], type] = {}

    def __getitem__(cls, width: int) -> type:
        if not isinstance(width, int) or width <= 0:
            raise TypeError(f"Width must be a positive integer, got {width!r}")

        key = (cls.__name__, width)
        if key not in cls._cache:
            cls._cache[key] = cls._make_type(width)
        return cls._cache[key]

    def _make_type(cls, width: int) -> type:
        raise NotImplementedError


class _UIntScalar(_ArithmeticMixin):
    """Instance of a UInt[N] value."""

    __slots__ = ("_value",)

    # These are overridden per-class by the generated subclass
    _width: int
    _mask: int
    _signed: bool = False

    def __init__(self, value: int = 0) -> None:
        self._value = int(value) & self._mask

    @property
    def value(self) -> int:
        return self._value

    @property
    def width(self) -> int:
        return self._width

    @property
    def signed(self) -> bool:
        return self._signed

    def _to_unsigned(self) -> int:
        """Return the raw unsigned bit pattern."""
        return self._value

    def _to_int(self) -> int:
        """Return the Python int interpretation (unsigned for UInt)."""
        return self._value

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._value == other._value

    def __hash__(self) -> int:
        return hash((type(self), self._value))

    def __repr__(self) -> str:
        return f"UInt[{self._width}]({self._value})"


class UInt(metaclass=_IntTypeMeta):
    """Unsigned arbitrary-width integer type.

    Usage:
        u8 = UInt[8]       # creates the UInt8 type
        x = u8(42)         # creates a UInt8 scalar with value 42
        x = UInt[8](42)    # shorthand for the above
    """

    # Class-level attributes so UInt[8].width works without an instance
    width: int
    signed: bool = False

    @classmethod
    def _make_type(cls, width: int) -> type:
        mask = (1 << width) - 1
        new_cls = type(
            f"UInt[{width}]",
            (_UIntScalar,),
            {
                "_width": width,
                "_mask": mask,
                "_signed": False,
                # Class-level properties so UInt[8].width works
                "width": width,
                "signed": False,
            },
        )
        return new_cls


class _SIntScalar(_ArithmeticMixin):
    """Instance of a SInt[N] value (two's complement signed)."""

    __slots__ = ("_value",)

    # These are overridden per-class by the generated subclass
    _width: int
    _mask: int
    _signed: bool = True
    _sign_bit: int
    _mod: int

    def __init__(self, value: int = 0) -> None:
        v = int(value) & self._mask
        if v >= self._sign_bit:
            v -= self._mod
        self._value = v

    @property
    def value(self) -> int:
        return self._value

    @property
    def width(self) -> int:
        return self._width

    @property
    def signed(self) -> bool:
        return self._signed

    def _to_unsigned(self) -> int:
        """Return the raw unsigned bit pattern."""
        return self._value & self._mask

    def _to_int(self) -> int:
        """Return the Python int interpretation (signed)."""
        return self._value

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._value == other._value

    def __hash__(self) -> int:
        return hash((type(self), self._value))

    def __repr__(self) -> str:
        return f"SInt[{self._width}]({self._value})"


class SInt(metaclass=_IntTypeMeta):
    """Signed arbitrary-width integer type (two's complement).

    Usage:
        s16 = SInt[16]       # creates the SInt16 type
        x = s16(-1)          # creates a SInt16 scalar with value -1
        x = SInt[16](1000)   # shorthand for the above
    """

    # Class-level attributes so SInt[16].width works without an instance
    width: int
    signed: bool = True

    @classmethod
    def _make_type(cls, width: int) -> type:
        mask = (1 << width) - 1
        sign_bit = 1 << (width - 1)
        mod = 1 << width
        new_cls = type(
            f"SInt[{width}]",
            (_SIntScalar,),
            {
                "_width": width,
                "_mask": mask,
                "_signed": True,
                "_sign_bit": sign_bit,
                "_mod": mod,
                # Class-level properties so SInt[16].width works
                "width": width,
                "signed": True,
            },
        )
        return new_cls


# --- Helper functions for arithmetic operators ---
# Defined after UInt/SInt so forward references resolve at call time.


def _is_rtl_type(obj):
    """Check if obj is a UInt or SInt scalar."""
    return isinstance(obj, (_UIntScalar, _SIntScalar))


def _make_result(value: int, width: int, signed: bool):
    """Create a result value of the given width and signedness."""
    if signed:
        return SInt[width](value)
    return UInt[width](value)
