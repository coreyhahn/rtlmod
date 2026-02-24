"""Arbitrary-width integer types for RTL modeling."""

from __future__ import annotations


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


class _UIntScalar:
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
