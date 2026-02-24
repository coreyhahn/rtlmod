# rtlmod Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python library providing arbitrary-width integer types with Verilog semantics for RTL modeling.

**Architecture:** Single package `rtlmod` with core types (`SInt[N]`, `UInt[N]`) backed by Python ints for scalars and numpy arrays for bulk operations. IO layer for test vector CSV/mem export and VCD comparison. Lightweight pipeline helper.

**Tech Stack:** Python 3.10+, numpy, pytest. Packaged with pyproject.toml, managed with uv.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `rtlmod/__init__.py`
- Create: `rtlmod/types.py`
- Create: `rtlmod/ops.py`
- Create: `rtlmod/array.py`
- Create: `rtlmod/pipeline.py`
- Create: `rtlmod/io/__init__.py`
- Create: `rtlmod/io/trace.py`
- Create: `rtlmod/io/vcd.py`
- Create: `tests/__init__.py`
- Create: `tests/test_types.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "rtlmod"
version = "0.1.0"
description = "Numpy-like arbitrary-width integer types for Python RTL modeling"
requires-python = ">=3.10"
dependencies = ["numpy"]

[project.optional-dependencies]
dev = ["pytest"]
```

**Step 2: Create package skeleton**

Create all files listed above as empty files (or with minimal docstrings). `rtlmod/__init__.py` should be:

```python
"""rtlmod - Numpy-like arbitrary-width integer types for Python RTL modeling."""
```

All other files should be empty or have a single-line module docstring.

**Step 3: Set up venv and install**

Run:
```bash
cd /home/cah/r2d2/code/fpga/python_models
uv venv
uv pip install -e ".[dev]"
```

**Step 4: Verify pytest runs**

Create `tests/test_types.py`:

```python
def test_placeholder():
    assert True
```

Run: `uv run pytest tests/ -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git init
git add pyproject.toml rtlmod/ tests/
git commit -m "chore: scaffold rtlmod package"
```

---

### Task 2: UInt Scalar - Construction and Masking

**Files:**
- Modify: `rtlmod/types.py`
- Modify: `rtlmod/__init__.py`
- Modify: `tests/test_types.py`

**Step 1: Write failing tests**

In `tests/test_types.py`:

```python
from rtlmod import UInt


class TestUIntConstruction:
    def test_type_creation(self):
        u8 = UInt[8]
        assert u8.width == 8
        assert u8.signed == False

    def test_scalar_creation(self):
        u8 = UInt[8]
        x = u8(42)
        assert x.value == 42
        assert x.width == 8
        assert x.signed == False

    def test_masking_on_construction(self):
        u8 = UInt[8]
        x = u8(256)  # overflow
        assert x.value == 0  # 256 & 0xFF == 0

    def test_masking_large_value(self):
        u8 = UInt[8]
        x = u8(0x1FF)  # 511
        assert x.value == 0xFF

    def test_negative_wraps(self):
        u8 = UInt[8]
        x = u8(-1)
        assert x.value == 255  # two's complement wrap

    def test_type_caching(self):
        """UInt[8] should return the same type object each time."""
        assert UInt[8] is UInt[8]

    def test_different_widths_different_types(self):
        assert UInt[8] is not UInt[16]

    def test_equality(self):
        u8 = UInt[8]
        assert u8(42) == u8(42)
        assert u8(42) != u8(43)

    def test_equality_cross_width(self):
        """Same value, different widths are not equal."""
        assert UInt[8](42) != UInt[16](42)

    def test_zero(self):
        u8 = UInt[8]
        x = u8(0)
        assert x.value == 0

    def test_max_value(self):
        u8 = UInt[8]
        x = u8(255)
        assert x.value == 255

    def test_wide_type(self):
        u258 = UInt[258]
        x = u258((1 << 258) - 1)
        assert x.value == (1 << 258) - 1

    def test_wide_masking(self):
        u258 = UInt[258]
        x = u258(1 << 258)
        assert x.value == 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_types.py::TestUIntConstruction -v`
Expected: FAIL with ImportError

**Step 3: Implement UInt in rtlmod/types.py**

```python
"""Core arbitrary-width integer types."""

from __future__ import annotations


class _IntTypeMeta(type):
    """Metaclass enabling SInt[N] and UInt[N] syntax."""

    _cache: dict[int, type] = {}

    def __getitem__(cls, width: int) -> type:
        if not isinstance(width, int) or width < 1:
            raise ValueError(f"Width must be a positive integer, got {width}")
        key = (cls._type_signed, width)
        if key not in cls._cache:
            cls._cache[key] = cls._make_type(width)
        return cls._cache[key]


class UInt(metaclass=_IntTypeMeta):
    """Unsigned integer type factory. Use UInt[N] to create an N-bit unsigned type."""

    _type_signed = False

    @classmethod
    def _make_type(cls, width: int) -> type:
        mask = (1 << width) - 1

        class _UIntN:
            __slots__ = ('_value',)

            def __init__(self, value: int = 0):
                self._value = int(value) & mask

            @property
            def value(self) -> int:
                return self._value

            @property
            def width(self) -> int:
                return width

            @property
            def signed(self) -> bool:
                return False

            def __eq__(self, other):
                if isinstance(other, _UIntN):
                    return self._value == other._value
                return NotImplemented

            def __hash__(self):
                return hash((width, False, self._value))

            def __repr__(self):
                return f"UInt[{width}]({self._value})"

        _UIntN.width = width
        _UIntN.signed = False
        _UIntN.__name__ = f"UInt[{width}]"
        _UIntN.__qualname__ = f"UInt[{width}]"
        return _UIntN

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
```

**Step 4: Export from __init__.py**

```python
"""rtlmod - Numpy-like arbitrary-width integer types for Python RTL modeling."""

from rtlmod.types import UInt
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_types.py::TestUIntConstruction -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add rtlmod/types.py rtlmod/__init__.py tests/test_types.py
git commit -m "feat: UInt[N] type with construction and masking"
```

---

### Task 3: SInt Scalar - Signed Construction and Masking

**Files:**
- Modify: `rtlmod/types.py`
- Modify: `rtlmod/__init__.py`
- Modify: `tests/test_types.py`

**Step 1: Write failing tests**

Add to `tests/test_types.py`:

```python
from rtlmod import SInt


class TestSIntConstruction:
    def test_type_creation(self):
        s16 = SInt[16]
        assert s16.width == 16
        assert s16.signed == True

    def test_positive_value(self):
        s16 = SInt[16]
        x = s16(1000)
        assert x.value == 1000

    def test_negative_value(self):
        s16 = SInt[16]
        x = s16(-1)
        assert x.value == -1

    def test_min_value(self):
        s16 = SInt[16]
        x = s16(-32768)
        assert x.value == -32768

    def test_max_value(self):
        s16 = SInt[16]
        x = s16(32767)
        assert x.value == 32767

    def test_positive_overflow_wraps(self):
        s16 = SInt[16]
        x = s16(32768)  # one past max
        assert x.value == -32768  # wraps to min

    def test_negative_overflow_wraps(self):
        s16 = SInt[16]
        x = s16(-32769)  # one past min
        assert x.value == 32767  # wraps to max

    def test_type_caching(self):
        assert SInt[16] is SInt[16]

    def test_sint_uint_different(self):
        assert SInt[8] is not UInt[8]

    def test_equality(self):
        s16 = SInt[16]
        assert s16(-1) == s16(-1)
        assert s16(0) != s16(1)

    def test_wide_signed(self):
        s258 = SInt[258]
        x = s258(-1)
        assert x.value == -1

    def test_wide_signed_max(self):
        s258 = SInt[258]
        max_val = (1 << 257) - 1
        x = s258(max_val)
        assert x.value == max_val

    def test_wide_signed_min(self):
        s258 = SInt[258]
        min_val = -(1 << 257)
        x = s258(min_val)
        assert x.value == min_val
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_types.py::TestSIntConstruction -v`
Expected: FAIL with ImportError

**Step 3: Implement SInt in rtlmod/types.py**

Add SInt class alongside UInt. The key difference is signed masking: after `& mask`, values >= `1 << (width - 1)` must be converted to negative by subtracting `1 << width`.

```python
class SInt(metaclass=_IntTypeMeta):
    """Signed integer type factory. Use SInt[N] to create an N-bit signed type."""

    _type_signed = True

    @classmethod
    def _make_type(cls, width: int) -> type:
        mask = (1 << width) - 1
        sign_bit = 1 << (width - 1)
        mod = 1 << width

        class _SIntN:
            __slots__ = ('_value',)

            def __init__(self, value: int = 0):
                v = int(value) & mask
                if v >= sign_bit:
                    v -= mod
                self._value = v

            @property
            def value(self) -> int:
                return self._value

            @property
            def width(self) -> int:
                return width

            @property
            def signed(self) -> bool:
                return True

            def __eq__(self, other):
                if isinstance(other, _SIntN):
                    return self._value == other._value
                return NotImplemented

            def __hash__(self):
                return hash((width, True, self._value))

            def __repr__(self):
                return f"SInt[{width}]({self._value})"

        _SIntN.width = width
        _SIntN.signed = True
        _SIntN.__name__ = f"SInt[{width}]"
        _SIntN.__qualname__ = f"SInt[{width}]"
        return _SIntN
```

**Step 4: Export from __init__.py**

```python
from rtlmod.types import UInt, SInt
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_types.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add rtlmod/types.py rtlmod/__init__.py tests/test_types.py
git commit -m "feat: SInt[N] type with signed construction and masking"
```

---

### Task 4: Arithmetic Operators with Verilog Width Rules

**Files:**
- Modify: `rtlmod/types.py`
- Create: `tests/test_arithmetic.py`

**Step 1: Write failing tests**

Create `tests/test_arithmetic.py`:

```python
from rtlmod import UInt, SInt


class TestAddition:
    def test_uint_add_widens(self):
        u8 = UInt[8]
        result = u8(200) + u8(100)
        assert result.width == 9  # grows by 1
        assert result.value == 300
        assert result.signed == False

    def test_sint_add_widens(self):
        s16 = SInt[16]
        result = s16(30000) + s16(30000)
        assert result.width == 17
        assert result.value == 60000
        assert result.signed == True

    def test_sint_add_negative(self):
        s16 = SInt[16]
        result = s16(-100) + s16(-200)
        assert result.width == 17
        assert result.value == -300

    def test_different_width_add(self):
        result = UInt[8](100) + UInt[16](1000)
        assert result.width == 17  # max(8,16) + 1
        assert result.value == 1100


class TestSubtraction:
    def test_uint_sub(self):
        u8 = UInt[8]
        result = u8(200) - u8(100)
        assert result.width == 9
        assert result.signed == True  # subtraction always produces signed
        assert result.value == 100

    def test_uint_sub_negative_result(self):
        u8 = UInt[8]
        result = u8(100) - u8(200)
        assert result.value == -100
        assert result.signed == True


class TestMultiplication:
    def test_uint_mul_doubles_width(self):
        u8 = UInt[8]
        result = u8(15) * u8(15)
        assert result.width == 16  # doubles
        assert result.value == 225

    def test_sint_mul(self):
        s16 = SInt[16]
        result = s16(-100) * s16(50)
        assert result.width == 32
        assert result.value == -5000

    def test_different_width_mul(self):
        result = UInt[8](10) * UInt[16](1000)
        assert result.width == 24  # 8 + 16
        assert result.value == 10000


class TestShifts:
    def test_left_shift_preserves_width(self):
        u8 = UInt[8]
        result = u8(1) << 4
        assert result.width == 8
        assert result.value == 16

    def test_left_shift_overflow_masks(self):
        u8 = UInt[8]
        result = u8(0xFF) << 1
        assert result.width == 8
        assert result.value == 0xFE

    def test_right_shift_preserves_width(self):
        u8 = UInt[8]
        result = u8(0xFF) >> 4
        assert result.width == 8
        assert result.value == 0x0F

    def test_signed_right_shift_arithmetic(self):
        s8 = SInt[8]
        result = s8(-4) >> 1
        assert result.width == 8
        assert result.value == -2  # arithmetic shift


class TestMixedSignedness:
    def test_uint_plus_sint(self):
        result = UInt[13](100) + SInt[16](50)
        assert result.signed == True
        assert result.width == 17  # max(13,16) + 1
        assert result.value == 150

    def test_uint_plus_sint_negative(self):
        result = UInt[8](10) + SInt[8](-20)
        assert result.signed == True
        assert result.value == -10


class TestBitwiseOps:
    def test_and(self):
        u8 = UInt[8]
        result = u8(0xF0) & u8(0x0F)
        assert result.value == 0
        assert result.width == 8

    def test_or(self):
        u8 = UInt[8]
        result = u8(0xF0) | u8(0x0F)
        assert result.value == 0xFF

    def test_xor(self):
        u8 = UInt[8]
        result = u8(0xFF) ^ u8(0x0F)
        assert result.value == 0xF0

    def test_invert(self):
        u8 = UInt[8]
        result = ~u8(0)
        assert result.value == 0xFF
        assert result.width == 8

    def test_neg(self):
        s8 = SInt[8]
        result = -s8(1)
        assert result.value == -1
        assert result.width == 8
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_arithmetic.py -v`
Expected: FAIL with TypeError/AttributeError

**Step 3: Implement operators**

Add `__add__`, `__radd__`, `__sub__`, `__rsub__`, `__mul__`, `__rmul__`, `__lshift__`, `__rshift__`, `__and__`, `__or__`, `__xor__`, `__invert__`, `__neg__` to both `_UIntN` and `_SIntN`.

The key logic for each operator:

- **Shared helper** `_resolve_binary(a, b)` returns `(a_val, b_val, result_width, result_signed)` applying Verilog promotion rules.
- **Addition/Subtraction:** result width = `max(wa, wb) + 1`. Subtraction always produces signed.
- **Multiplication:** result width = `wa + wb`.
- **Shifts:** result width = same as left operand. Right operand is a plain int.
- **Bitwise:** result width = `max(wa, wb)`. Signedness follows Verilog rules (signed only if both signed).
- **Invert:** same width and signedness.
- **Negate:** same width, signed.

Create the result using `SInt[w](val)` or `UInt[w](val)` which handles masking automatically.

Implementation approach: add a module-level helper function that both UInt and SInt generated classes call. The generated classes store `_width`, `_signed`, and `_mask` as closure variables. Operators call the helper, which returns a new instance of the appropriate result type.

```python
def _make_result(value: int, width: int, signed: bool):
    """Create a result value of the given width and signedness."""
    if signed:
        return SInt[width](value)
    return UInt[width](value)


def _binary_result_type(a, b):
    """Determine result width and signedness for binary ops (Verilog rules)."""
    a_w, a_s = a.width, a.signed
    b_w, b_s = b.width, b.signed
    # If either is signed, result is signed (and unsigned operand is widened by 1)
    if a_s and b_s:
        return max(a_w, b_w), True
    elif a_s or b_s:
        return max(a_w, b_w), True
    else:
        return max(a_w, b_w), False
```

Then in each generated class, add:

```python
def __add__(self, other):
    w, s = _binary_result_type(self, other)
    return _make_result(self._to_int() + other._to_int(), w + 1, s)

def __sub__(self, other):
    w, _ = _binary_result_type(self, other)
    return _make_result(self._to_int() - other._to_int(), w + 1, True)  # sub always signed

def __mul__(self, other):
    _, s = _binary_result_type(self, other)
    return _make_result(self._to_int() * other._to_int(), self.width + other.width, s)

def __lshift__(self, n):
    return type(self)(self._value << int(n))  # masks via constructor

def __rshift__(self, n):
    return type(self)(self._to_int() >> int(n))  # arithmetic for signed

def __and__(self, other):
    w, s = _binary_result_type(self, other)
    return _make_result(self._to_unsigned() & other._to_unsigned(), w, s)

def __or__(self, other):
    w, s = _binary_result_type(self, other)
    return _make_result(self._to_unsigned() | other._to_unsigned(), w, s)

def __xor__(self, other):
    w, s = _binary_result_type(self, other)
    return _make_result(self._to_unsigned() ^ other._to_unsigned(), w, s)

def __invert__(self):
    return type(self)(~self._to_unsigned())  # masks via constructor

def __neg__(self):
    if self.signed:
        return type(self)(-self._value)
    return SInt[self.width](-self._value)
```

Note: `_to_int()` returns the signed Python int for signed types, unsigned for unsigned. `_to_unsigned()` returns the raw unsigned bit pattern. These are needed so arithmetic on signed values uses Python's signed semantics (especially for `>>`).

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_arithmetic.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add rtlmod/types.py tests/test_arithmetic.py
git commit -m "feat: arithmetic operators with Verilog width rules"
```

---

### Task 5: Bit Slicing, Reductions, and Concat

**Files:**
- Modify: `rtlmod/types.py`
- Modify: `rtlmod/ops.py`
- Modify: `rtlmod/__init__.py`
- Create: `tests/test_ops.py`

**Step 1: Write failing tests**

Create `tests/test_ops.py`:

```python
from rtlmod import UInt, SInt, concat


class TestBitSlice:
    def test_single_bit(self):
        u8 = UInt[8]
        x = u8(0b10110100)
        assert x[0].value == 0   # LSB
        assert x[7].value == 1   # MSB
        assert x[2].value == 1
        assert x[0].width == 1

    def test_bit_range(self):
        u8 = UInt[8]
        x = u8(0xAB)  # 10101011
        low = x[3:0]   # lower nibble = 1011 = 0xB
        assert low.value == 0xB
        assert low.width == 4

    def test_bit_range_upper(self):
        u8 = UInt[8]
        x = u8(0xAB)
        high = x[7:4]  # upper nibble = 1010 = 0xA
        assert high.value == 0xA
        assert high.width == 4

    def test_slice_always_unsigned(self):
        s16 = SInt[16]
        x = s16(-1)  # all 1s
        low = x[7:0]
        assert low.value == 0xFF
        assert low.signed == False

    def test_wide_slice(self):
        u258 = UInt[258]
        x = u258((1 << 200) | 0xFF)
        low = x[7:0]
        assert low.value == 0xFF
        high_bit = x[200]
        assert high_bit.value == 1


class TestReductions:
    def test_xor_reduce(self):
        u8 = UInt[8]
        assert u8(0b11001100).xor_reduce().value == 0  # even number of 1s
        assert u8(0b11001101).xor_reduce().value == 1  # odd number of 1s

    def test_and_reduce(self):
        u8 = UInt[8]
        assert u8(0xFF).and_reduce().value == 1
        assert u8(0xFE).and_reduce().value == 0

    def test_or_reduce(self):
        u8 = UInt[8]
        assert u8(0x00).or_reduce().value == 0
        assert u8(0x01).or_reduce().value == 1

    def test_reduction_returns_uint1(self):
        u8 = UInt[8]
        r = u8(0xFF).xor_reduce()
        assert r.width == 1
        assert r.signed == False


class TestConcat:
    def test_two_values(self):
        u8 = UInt[8]
        result = concat(u8(0xAB), u8(0xCD))
        assert result.width == 16
        assert result.value == 0xABCD

    def test_three_values(self):
        u4 = UInt[4]
        result = concat(u4(0xA), u4(0xB), u4(0xC))
        assert result.width == 12
        assert result.value == 0xABC

    def test_mixed_widths(self):
        result = concat(UInt[4](0xF), UInt[8](0xAB))
        assert result.width == 12
        assert result.value == 0xFAB

    def test_concat_always_unsigned(self):
        result = concat(SInt[8](-1), SInt[8](-1))
        assert result.signed == False
        assert result.value == 0xFFFF

    def test_single_value(self):
        result = concat(UInt[8](0xAB))
        assert result.value == 0xAB
        assert result.width == 8
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ops.py -v`
Expected: FAIL

**Step 3: Implement bit slicing on types**

Add `__getitem__` to both `_UIntN` and `_SIntN`:

```python
def __getitem__(self, key):
    raw = self._to_unsigned()
    if isinstance(key, int):
        # Single bit
        return UInt[1]((raw >> key) & 1)
    elif isinstance(key, slice):
        # Verilog-style: val[msb:lsb]
        msb = key.start
        lsb = key.stop
        if msb is None or lsb is None:
            raise ValueError("Bit slice requires both MSB and LSB: val[msb:lsb]")
        w = msb - lsb + 1
        return UInt[w]((raw >> lsb) & ((1 << w) - 1))
    raise TypeError(f"Invalid index type: {type(key)}")
```

**Step 4: Implement reductions on types**

Add to both generated classes:

```python
def xor_reduce(self):
    v = self._to_unsigned()
    r = 0
    for i in range(self.width):
        r ^= (v >> i) & 1
    return UInt[1](r)

def and_reduce(self):
    mask = (1 << self.width) - 1
    return UInt[1](1 if (self._to_unsigned() & mask) == mask else 0)

def or_reduce(self):
    return UInt[1](1 if self._to_unsigned() != 0 else 0)
```

**Step 5: Implement concat in rtlmod/ops.py**

```python
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
```

Export from `rtlmod/__init__.py`:

```python
from rtlmod.ops import concat
```

**Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_ops.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add rtlmod/types.py rtlmod/ops.py rtlmod/__init__.py tests/test_ops.py
git commit -m "feat: bit slicing, reductions, and concat"
```

---

### Task 6: Resize, Sign Extend, and Display

**Files:**
- Modify: `rtlmod/types.py`
- Create: `tests/test_resize_display.py`

**Step 1: Write failing tests**

Create `tests/test_resize_display.py`:

```python
from rtlmod import UInt, SInt


class TestResize:
    def test_truncate_unsigned(self):
        u16 = UInt[16]
        result = u16(0xABCD).resize(8)
        assert result.width == 8
        assert result.value == 0xCD  # lower 8 bits
        assert result.signed == False

    def test_truncate_signed(self):
        s16 = SInt[16]
        result = s16(0x7FFF).resize(8)
        assert result.width == 8
        assert result.signed == True

    def test_widen_unsigned(self):
        u8 = UInt[8]
        result = u8(0xAB).resize(16)
        assert result.width == 16
        assert result.value == 0xAB  # zero-extended

    def test_widen_signed_positive(self):
        s8 = SInt[8]
        result = s8(42).resize(16)
        assert result.width == 16
        assert result.value == 42

    def test_widen_signed_negative(self):
        s8 = SInt[8]
        result = s8(-1).resize(16)
        assert result.width == 16
        assert result.value == -1  # sign-extended

    def test_saturate_unsigned(self):
        u16 = UInt[16]
        result = u16(300).resize(8, round='saturate')
        assert result.value == 255  # clamped to max

    def test_saturate_signed_positive(self):
        s16 = SInt[16]
        result = s16(200).resize(8, round='saturate')
        assert result.value == 127

    def test_saturate_signed_negative(self):
        s16 = SInt[16]
        result = s16(-200).resize(8, round='saturate')
        assert result.value == -128


class TestSignExtend:
    def test_sign_extend_positive(self):
        u8 = UInt[8]
        result = u8(42).sign_extend(16)
        assert result.width == 16
        assert result.signed == True
        assert result.value == 42

    def test_sign_extend_with_msb_set(self):
        u8 = UInt[8]
        result = u8(0x80).sign_extend(16)
        assert result.value == -128  # MSB of u8 becomes sign bit

    def test_sign_extend_signed(self):
        s8 = SInt[8]
        result = s8(-5).sign_extend(16)
        assert result.value == -5
        assert result.width == 16


class TestDisplay:
    def test_uint_str(self):
        u8 = UInt[8]
        assert str(u8(42)) == "u8'd42"

    def test_sint_str(self):
        s16 = SInt[16]
        assert str(s16(-1)) == "s16'd-1"

    def test_uint_hex(self):
        u8 = UInt[8]
        assert UInt[8](0xAB).hex == "8'h00ab"

    def test_sint_hex(self):
        s16 = SInt[16]
        assert SInt[16](-1).hex == "16'shffff"

    def test_uint_bin(self):
        u4 = UInt[4]
        assert u4(0b1010).bin == "4'b1010"

    def test_sint_bin(self):
        s4 = SInt[4]
        assert s4(-1).bin == "4'sb1111"

    def test_repr(self):
        assert repr(UInt[8](42)) == "UInt[8](42)"
        assert repr(SInt[16](-1)) == "SInt[16](-1)"

    def test_wide_hex(self):
        u258 = UInt[258]
        x = u258(0xFF)
        # Should be zero-padded to full width in hex
        h = x.hex
        assert h.startswith("258'h")
        assert h.endswith("ff")
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_resize_display.py -v`
Expected: FAIL

**Step 3: Implement resize and sign_extend**

Add to both generated classes:

```python
def resize(self, new_width: int, round: str = 'truncate'):
    """Resize to new_width. round='truncate' (default) or 'saturate'."""
    if round == 'truncate':
        if self.signed:
            return SInt[new_width](self._value)
        return UInt[new_width](self._value)
    elif round == 'saturate':
        if self.signed:
            max_val = (1 << (new_width - 1)) - 1
            min_val = -(1 << (new_width - 1))
            clamped = max(min_val, min(max_val, self._value))
            return SInt[new_width](clamped)
        else:
            max_val = (1 << new_width) - 1
            clamped = min(max_val, self._value)
            return UInt[new_width](clamped)
    raise ValueError(f"Unknown round mode: {round}")

def sign_extend(self, new_width: int):
    """Sign-extend to new_width. Returns SInt."""
    # Get value interpreting current MSB as sign
    raw = self._to_unsigned()
    if raw >= (1 << (self.width - 1)):
        val = raw - (1 << self.width)
    else:
        val = raw
    return SInt[new_width](val)
```

**Step 4: Implement display methods**

Add `__str__`, `hex` property, `bin` property to both classes:

```python
def __str__(self):
    prefix = 's' if self.signed else 'u'
    return f"{prefix}{self.width}'d{self._value}"

@property
def hex(self):
    raw = self._to_unsigned()
    # Number of hex digits needed for full width
    hex_digits = (self.width + 3) // 4
    prefix = "sh" if self.signed else "h"
    return f"{self.width}'{prefix}{raw:0{hex_digits}x}"

@property
def bin(self):
    raw = self._to_unsigned()
    prefix = "sb" if self.signed else "b"
    return f"{self.width}'{prefix}{raw:0{self.width}b}"
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_resize_display.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add rtlmod/types.py tests/test_resize_display.py
git commit -m "feat: resize, sign_extend, and display formatting"
```

---

### Task 7: Numpy-Backed Arrays

**Files:**
- Modify: `rtlmod/array.py`
- Modify: `rtlmod/types.py` (add `.array`, `.random`, `.arange`, `.from_numpy` class methods)
- Create: `tests/test_array.py`

**Step 1: Write failing tests**

Create `tests/test_array.py`:

```python
import numpy as np
from rtlmod import UInt, SInt


class TestArrayCreation:
    def test_zeros(self):
        u8 = UInt[8]
        arr = u8.array(shape=(4, 4))
        assert arr.shape == (4, 4)
        assert arr.width == 8
        assert arr.signed == False

    def test_from_list(self):
        u8 = UInt[8]
        arr = u8.array([1, 2, 3, 4])
        assert arr.shape == (4,)
        assert arr[0] == u8(1)
        assert arr[3] == u8(4)

    def test_masking_on_creation(self):
        u8 = UInt[8]
        arr = u8.array([256, 257, 258])
        assert arr[0] == u8(0)
        assert arr[1] == u8(1)
        assert arr[2] == u8(2)

    def test_random(self):
        u8 = UInt[8]
        arr = u8.random(shape=(100,))
        assert arr.shape == (100,)
        # All values in valid range
        raw = arr.to_numpy()
        assert np.all(raw >= 0)
        assert np.all(raw <= 255)

    def test_arange(self):
        u8 = UInt[8]
        arr = u8.arange(0, 10)
        assert arr.shape == (10,)
        assert arr[0] == u8(0)
        assert arr[9] == u8(9)

    def test_signed_array(self):
        s16 = SInt[16]
        arr = s16.array([100, -200, 300, -400])
        assert arr[1] == s16(-200)
        assert arr.signed == True


class TestArrayArithmetic:
    def test_add(self):
        u8 = UInt[8]
        a = u8.array([200, 100])
        b = u8.array([100, 200])
        result = a + b
        assert result.width == 9
        assert result[0] == UInt[9](300)
        assert result[1] == UInt[9](300)

    def test_mul(self):
        u8 = UInt[8]
        a = u8.array([10, 20])
        b = u8.array([30, 40])
        result = a * b
        assert result.width == 16
        assert result[0] == UInt[16](300)
        assert result[1] == UInt[16](800)

    def test_shift_preserves_width(self):
        u8 = UInt[8]
        a = u8.array([0xFF, 0x0F])
        result = a << 1
        assert result.width == 8
        assert result[0] == u8(0xFE)

    def test_scalar_broadcast(self):
        u8 = UInt[8]
        a = u8.array([10, 20, 30])
        result = a + u8(5)
        assert result[0] == UInt[9](15)
        assert result[2] == UInt[9](35)


class TestArrayNumpyInterop:
    def test_to_numpy(self):
        u8 = UInt[8]
        arr = u8.array([1, 2, 3])
        raw = arr.to_numpy()
        assert isinstance(raw, np.ndarray)
        assert list(raw) == [1, 2, 3]

    def test_from_numpy(self):
        u8 = UInt[8]
        raw = np.array([1, 2, 3], dtype=np.int16)
        arr = u8.from_numpy(raw)
        assert arr[0] == u8(1)
        assert arr.width == 8

    def test_from_numpy_masks(self):
        u8 = UInt[8]
        raw = np.array([256, 257], dtype=np.int32)
        arr = u8.from_numpy(raw)
        assert arr[0] == u8(0)
        assert arr[1] == u8(1)


class TestArrayIndexing:
    def test_single_element(self):
        u8 = UInt[8]
        arr = u8.array([10, 20, 30])
        x = arr[1]
        assert x == u8(20)
        assert isinstance(x, type(u8(0)))

    def test_slice(self):
        u8 = UInt[8]
        arr = u8.array([10, 20, 30, 40, 50])
        sub = arr[1:3]
        assert sub.shape == (2,)
        assert sub[0] == u8(20)

    def test_2d_indexing(self):
        u8 = UInt[8]
        arr = u8.array(shape=(4, 4))
        assert arr[2, 3] == u8(0)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_array.py -v`
Expected: FAIL

**Step 3: Implement RtlArray in rtlmod/array.py**

```python
"""Numpy-backed arbitrary-width integer arrays."""

from __future__ import annotations
import numpy as np


def _backing_dtype(width: int, signed: bool):
    """Choose the smallest numpy dtype that fits the given width."""
    if signed:
        if width <= 8: return np.int8
        if width <= 16: return np.int16
        if width <= 32: return np.int32
        if width <= 64: return np.int64
        return object  # Python ints for very wide values
    else:
        if width <= 8: return np.uint8
        if width <= 16: return np.uint16
        if width <= 32: return np.uint32
        if width <= 64: return np.uint64
        return object


class RtlArray:
    """Numpy-backed array with arbitrary bit-width and Verilog masking semantics."""

    __slots__ = ('_data', '_width', '_signed', '_mask', '_sign_bit', '_mod')

    def __init__(self, data: np.ndarray, width: int, signed: bool):
        self._data = data
        self._width = width
        self._signed = signed
        self._mask = (1 << width) - 1
        self._sign_bit = 1 << (width - 1)
        self._mod = 1 << width

    @property
    def shape(self):
        return self._data.shape

    @property
    def width(self):
        return self._width

    @property
    def signed(self):
        return self._signed

    def _mask_data(self, data: np.ndarray, width: int, signed: bool) -> RtlArray:
        """Create new RtlArray with masking applied."""
        mask = (1 << width) - 1
        if data.dtype == object:
            masked = np.vectorize(lambda v: int(v) & mask)(data)
            if signed:
                sign_bit = 1 << (width - 1)
                mod = 1 << width
                masked = np.vectorize(lambda v: v - mod if v >= sign_bit else v)(masked)
        else:
            masked = data.astype(np.int64) & mask
            if signed:
                sign_bit = 1 << (width - 1)
                mod = 1 << width
                masked = np.where(masked >= sign_bit, masked - mod, masked)
        return RtlArray(masked, width, signed)

    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(result, np.ndarray):
            return RtlArray(result, self._width, self._signed)
        # Scalar - return typed scalar
        from rtlmod.types import SInt, UInt
        if self._signed:
            return SInt[self._width](int(result))
        return UInt[self._width](int(result))

    def __eq__(self, other):
        if isinstance(other, RtlArray):
            return np.array_equal(self._data, other._data) and self._width == other._width
        return NotImplemented

    def to_numpy(self) -> np.ndarray:
        return self._data.copy()

    # Arithmetic operators - follow same Verilog rules as scalars
    def __add__(self, other):
        if isinstance(other, RtlArray):
            w = max(self._width, other._width)
            s = self._signed or other._signed
            return self._mask_data(self._data.astype(np.int64) + other._data.astype(np.int64), w + 1, s)
        # Scalar broadcast
        w = max(self._width, other.width)
        s = self._signed or other.signed
        return self._mask_data(self._data.astype(np.int64) + other.value, w + 1, s)

    def __mul__(self, other):
        if isinstance(other, RtlArray):
            s = self._signed or other._signed
            return self._mask_data(self._data.astype(np.int64) * other._data.astype(np.int64),
                                   self._width + other._width, s)
        s = self._signed or other.signed
        return self._mask_data(self._data.astype(np.int64) * other.value,
                               self._width + other.width, s)

    def __lshift__(self, n):
        return self._mask_data(self._data.astype(np.int64) << int(n), self._width, self._signed)

    def __rshift__(self, n):
        return self._mask_data(self._data.astype(np.int64) >> int(n), self._width, self._signed)
```

**Step 4: Add factory methods to type classes in rtlmod/types.py**

Add class-level methods `array`, `random`, `arange`, `from_numpy` to the generated type classes. These create `RtlArray` instances:

```python
@classmethod
def array(cls_inner, data=None, shape=None):
    from rtlmod.array import RtlArray, _backing_dtype
    dtype = _backing_dtype(width, is_signed)
    if data is not None:
        np_data = np.array(data, dtype=np.int64)
    elif shape is not None:
        np_data = np.zeros(shape, dtype=np.int64)
    else:
        raise ValueError("Provide data or shape")
    arr = RtlArray(np_data, width, is_signed)
    return arr._mask_data(np_data, width, is_signed)

@classmethod
def random(cls_inner, shape):
    from rtlmod.array import RtlArray
    if is_signed:
        lo, hi = -(1 << (width - 1)), (1 << (width - 1))
    else:
        lo, hi = 0, 1 << width
    np_data = np.random.randint(lo, hi, size=shape, dtype=np.int64)
    return RtlArray(np_data, width, is_signed)

@classmethod
def arange(cls_inner, start, stop):
    from rtlmod.array import RtlArray
    np_data = np.arange(start, stop, dtype=np.int64)
    arr = RtlArray(np_data, width, is_signed)
    return arr._mask_data(np_data, width, is_signed)

@classmethod
def from_numpy(cls_inner, np_array):
    from rtlmod.array import RtlArray
    np_data = np_array.astype(np.int64)
    arr = RtlArray(np_data, width, is_signed)
    return arr._mask_data(np_data, width, is_signed)
```

Where `is_signed` is `True` for SInt, `False` for UInt (captured from the closure in `_make_type`).

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_array.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add rtlmod/array.py rtlmod/types.py tests/test_array.py
git commit -m "feat: numpy-backed RtlArray with Verilog arithmetic"
```

---

### Task 8: Trace Recording and CSV Export/Import

**Files:**
- Modify: `rtlmod/io/trace.py`
- Modify: `rtlmod/io/__init__.py`
- Create: `tests/test_trace.py`

**Step 1: Write failing tests**

Create `tests/test_trace.py`:

```python
import os
import tempfile
from rtlmod import UInt, SInt
from rtlmod.io import Trace


class TestTraceRecord:
    def test_record_and_length(self):
        trace = Trace()
        u8 = UInt[8]
        for i in range(10):
            trace.record(x=u8(i), y=u8(i * 2))
        assert len(trace) == 10

    def test_record_signals(self):
        trace = Trace()
        u8 = UInt[8]
        trace.record(a=u8(42), b=u8(99))
        assert trace.signals == ['a', 'b']


class TestTraceCSV:
    def test_to_csv_hex(self):
        trace = Trace()
        u8 = UInt[8]
        trace.record(x=u8(0xAB), y=u8(0xCD))
        trace.record(x=u8(0x12), y=u8(0x34))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        try:
            trace.to_csv(path)
            with open(path) as f:
                lines = f.readlines()
            assert 'cycle' in lines[0]
            assert 'x' in lines[0]
            assert 'y' in lines[0]
            assert len(lines) == 3  # header + 2 data rows
        finally:
            os.unlink(path)

    def test_to_csv_decimal(self):
        trace = Trace()
        s16 = SInt[16]
        trace.record(val=s16(-100))
        trace.record(val=s16(200))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        try:
            trace.to_csv(path, fmt='decimal')
            with open(path) as f:
                content = f.read()
            assert '-100' in content
            assert '200' in content
        finally:
            os.unlink(path)

    def test_roundtrip_csv(self):
        trace = Trace()
        u8 = UInt[8]
        s16 = SInt[16]
        for i in range(5):
            trace.record(a=u8(i * 10), b=s16(i * -100))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        try:
            trace.to_csv(path)
            loaded = Trace.from_csv(path, types={'a': UInt[8], 'b': SInt[16]})
            assert len(loaded) == 5
            assert loaded[0].a == u8(0)
            assert loaded[4].b == s16(-400)
        finally:
            os.unlink(path)


class TestTraceIteration:
    def test_iterate(self):
        trace = Trace()
        u8 = UInt[8]
        trace.record(x=u8(10))
        trace.record(x=u8(20))
        values = [cycle.x.value for cycle in trace]
        assert values == [10, 20]

    def test_index(self):
        trace = Trace()
        u8 = UInt[8]
        trace.record(x=u8(10), y=u8(20))
        trace.record(x=u8(30), y=u8(40))
        assert trace[1].x == u8(30)
        assert trace[1].y == u8(40)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_trace.py -v`
Expected: FAIL

**Step 3: Implement Trace in rtlmod/io/trace.py**

```python
"""Trace recorder for cycle-by-cycle signal capture."""

from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace


class Trace:
    """Records named signals per cycle. Primary IO object for test vector export."""

    def __init__(self):
        self._signals: list[str] = []
        self._cycles: list[dict] = []
        self._types: dict[str, type] = {}

    @property
    def signals(self) -> list[str]:
        return list(self._signals)

    def __len__(self):
        return len(self._cycles)

    def record(self, **kwargs):
        """Record one cycle of signal values."""
        if not self._signals:
            self._signals = list(kwargs.keys())
            self._types = {name: type(val) for name, val in kwargs.items()}
        self._cycles.append(dict(kwargs))

    def __getitem__(self, index):
        return SimpleNamespace(**self._cycles[index])

    def __iter__(self):
        for cycle_data in self._cycles:
            yield SimpleNamespace(**cycle_data)

    def _format_value(self, val, fmt: str) -> str:
        if fmt == 'hex':
            raw = val._to_unsigned()
            hex_digits = (val.width + 3) // 4
            return f"{raw:0{hex_digits}x}"
        elif fmt == 'decimal':
            return str(val.value)
        elif fmt == 'binary':
            raw = val._to_unsigned()
            return f"{raw:0{val.width}b}"
        raise ValueError(f"Unknown format: {fmt}")

    def to_csv(self, path: str, fmt: str = 'hex'):
        """Export trace to CSV file."""
        p = Path(path)
        with open(p, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cycle'] + self._signals)
            for i, cycle_data in enumerate(self._cycles):
                row = [i]
                for sig in self._signals:
                    row.append(self._format_value(cycle_data[sig], fmt))
                writer.writerow(row)

    @classmethod
    def from_csv(cls, path: str, types: dict[str, type]) -> Trace:
        """Load trace from CSV file."""
        trace = Trace()
        p = Path(path)
        with open(p, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kwargs = {}
                for name, typ in types.items():
                    raw = row[name].strip()
                    # Detect hex (default export format)
                    try:
                        val = int(raw, 16)
                    except ValueError:
                        val = int(raw)
                    kwargs[name] = typ(val)
                trace.record(**kwargs)
        return trace
```

**Step 4: Export from rtlmod/io/__init__.py**

```python
from rtlmod.io.trace import Trace
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_trace.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add rtlmod/io/trace.py rtlmod/io/__init__.py tests/test_trace.py
git commit -m "feat: Trace recording with CSV export/import"
```

---

### Task 9: .mem File Export

**Files:**
- Modify: `rtlmod/io/trace.py`
- Create: `tests/test_mem.py`

**Step 1: Write failing tests**

Create `tests/test_mem.py`:

```python
import os
import tempfile
from pathlib import Path
from rtlmod import UInt, SInt
from rtlmod.io import Trace


class TestMemExport:
    def test_mem_directory(self):
        trace = Trace()
        u8 = UInt[8]
        trace.record(a=u8(0xAB), b=u8(0xCD))
        trace.record(a=u8(0x12), b=u8(0x34))

        with tempfile.TemporaryDirectory() as tmpdir:
            trace.to_mem(tmpdir)
            a_path = Path(tmpdir) / "a.mem"
            b_path = Path(tmpdir) / "b.mem"
            assert a_path.exists()
            assert b_path.exists()

            a_lines = a_path.read_text().strip().split('\n')
            assert a_lines == ['ab', '12']

            b_lines = b_path.read_text().strip().split('\n')
            assert b_lines == ['cd', '34']

    def test_mem_single_signal(self):
        trace = Trace()
        u8 = UInt[8]
        trace.record(x=u8(0xFF), y=u8(0x00))
        trace.record(x=u8(0xAA), y=u8(0x55))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.mem', delete=False) as f:
            path = f.name
        try:
            trace.to_mem(path, signals=['x'])
            lines = Path(path).read_text().strip().split('\n')
            assert lines == ['ff', 'aa']
        finally:
            os.unlink(path)

    def test_mem_wide_signal(self):
        trace = Trace()
        u258 = UInt[258]
        trace.record(f=u258(0xFF))

        with tempfile.TemporaryDirectory() as tmpdir:
            trace.to_mem(tmpdir)
            content = (Path(tmpdir) / "f.mem").read_text().strip()
            # Should be zero-padded hex, $readmemh compatible
            assert content.endswith('ff')
            # Width 258 = 65 hex digits
            assert len(content) == 65

    def test_mem_signed_signal(self):
        trace = Trace()
        s16 = SInt[16]
        trace.record(val=s16(-1))

        with tempfile.TemporaryDirectory() as tmpdir:
            trace.to_mem(tmpdir)
            content = (Path(tmpdir) / "val.mem").read_text().strip()
            assert content == 'ffff'  # unsigned bit pattern
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mem.py -v`
Expected: FAIL

**Step 3: Implement to_mem on Trace**

Add to `rtlmod/io/trace.py`:

```python
def to_mem(self, path: str, signals: list[str] | None = None):
    """Export to .mem files ($readmemh compatible).

    If path is a directory: creates one file per signal (a.mem, b.mem, ...).
    If path is a file and signals is provided: writes those signals to one file.
    """
    p = Path(path)
    sigs = signals or self._signals

    if p.is_dir() or (not p.suffix and not p.exists()):
        # Directory mode: one file per signal
        p.mkdir(parents=True, exist_ok=True)
        for sig in sigs:
            sig_path = p / f"{sig}.mem"
            self._write_mem_file(sig_path, sig)
    else:
        # Single file mode
        if len(sigs) == 1:
            self._write_mem_file(p, sigs[0])
        else:
            raise ValueError("Single file export requires exactly one signal")

def _write_mem_file(self, path: Path, signal: str):
    """Write one signal to a $readmemh-compatible .mem file."""
    with open(path, 'w') as f:
        for cycle_data in self._cycles:
            val = cycle_data[signal]
            raw = val._to_unsigned()
            hex_digits = (val.width + 3) // 4
            f.write(f"{raw:0{hex_digits}x}\n")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mem.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add rtlmod/io/trace.py tests/test_mem.py
git commit -m "feat: .mem file export for $readmemh"
```

---

### Task 10: VCD Reader and RTL Comparison

**Files:**
- Modify: `rtlmod/io/vcd.py`
- Modify: `rtlmod/io/__init__.py`
- Create: `tests/test_vcd.py`
- Create: `tests/fixtures/simple.vcd` (test fixture)

**Step 1: Create a minimal VCD test fixture**

Create `tests/fixtures/simple.vcd`:

```vcd
$timescale 1ns $end
$scope module tb $end
$scope module dut $end
$var wire 8 ! x [7:0] $end
$var wire 8 " y [7:0] $end
$upscope $end
$upscope $end
$enddefinitions $end
$dumpvars
b00000000 !
b00000000 "
$end
#0
b00001010 !
b00010100 "
#10
b00010100 !
b00101000 "
#20
b00011110 !
b00111100 "
```

**Step 2: Write failing tests**

Create `tests/test_vcd.py`:

```python
import os
import tempfile
from pathlib import Path
from rtlmod import UInt, SInt
from rtlmod.io import Trace, VCD

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestVCDRead:
    def test_read_signals(self):
        vcd = VCD.read(FIXTURE_DIR / "simple.vcd")
        u8 = UInt[8]
        x = vcd.signal("tb.dut.x", as_type=u8)
        assert len(x) >= 3
        assert x[0] == u8(10)
        assert x[1] == u8(20)
        assert x[2] == u8(30)

    def test_read_multiple_signals(self):
        vcd = VCD.read(FIXTURE_DIR / "simple.vcd")
        u8 = UInt[8]
        x = vcd.signal("tb.dut.x", as_type=u8)
        y = vcd.signal("tb.dut.y", as_type=u8)
        assert x[0].value == 10
        assert y[0].value == 20


class TestVCDCompare:
    def test_compare_matching(self):
        # Build a trace that matches the VCD
        trace = Trace()
        u8 = UInt[8]
        trace.record(x=u8(10), y=u8(20))
        trace.record(x=u8(20), y=u8(40))
        trace.record(x=u8(30), y=u8(60))

        vcd = VCD.read(FIXTURE_DIR / "simple.vcd")
        diff = trace.compare(vcd, mapping={
            'x': 'tb.dut.x',
            'y': 'tb.dut.y',
        })
        assert diff.passed

    def test_compare_mismatch(self):
        trace = Trace()
        u8 = UInt[8]
        trace.record(x=u8(10))
        trace.record(x=u8(99))  # wrong value

        vcd = VCD.read(FIXTURE_DIR / "simple.vcd")
        diff = trace.compare(vcd, mapping={'x': 'tb.dut.x'})
        assert not diff.passed
        assert diff.first_error.cycle == 1
        assert diff.first_error.signal == 'x'

    def test_compare_summary(self):
        trace = Trace()
        u8 = UInt[8]
        trace.record(x=u8(10))
        trace.record(x=u8(99))

        vcd = VCD.read(FIXTURE_DIR / "simple.vcd")
        diff = trace.compare(vcd, mapping={'x': 'tb.dut.x'})
        summary = diff.summary()
        assert 'FAIL' in summary or 'mismatch' in summary.lower()
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_vcd.py -v`
Expected: FAIL

**Step 4: Implement VCD reader in rtlmod/io/vcd.py**

```python
"""VCD (Value Change Dump) reader for RTL simulation comparison."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CompareError:
    cycle: int
    signal: str
    expected: object
    actual: object


@dataclass
class CompareResult:
    passed: bool
    mismatches: list[CompareError] = field(default_factory=list)

    @property
    def first_error(self) -> CompareError | None:
        return self.mismatches[0] if self.mismatches else None

    def summary(self) -> str:
        if self.passed:
            return f"PASS: all cycles matched"
        return (f"FAIL: {len(self.mismatches)} mismatch(es). "
                f"First at cycle {self.first_error.cycle}, "
                f"signal '{self.first_error.signal}': "
                f"expected {self.first_error.expected}, "
                f"got {self.first_error.actual}")


class VCD:
    """VCD file reader. Extracts signal values at each timestamp."""

    def __init__(self, signals: dict[str, list], timestamps: list[int]):
        self._signals = signals  # {hierarchical_name: [values at each timestamp]}
        self._timestamps = timestamps

    @classmethod
    def read(cls, path: str | Path) -> VCD:
        """Parse a VCD file and return a VCD object."""
        path = Path(path)
        text = path.read_text()

        # Parse header: map var codes to hierarchical names
        var_map = {}  # code -> full_name
        scope_stack = []
        lines = text.split('\n')
        i = 0
        in_defs = True

        while i < len(lines) and in_defs:
            line = lines[i].strip()
            if '$scope' in line:
                parts = line.split()
                scope_idx = parts.index('$scope')
                scope_stack.append(parts[scope_idx + 2])
            elif '$upscope' in line:
                if scope_stack:
                    scope_stack.pop()
            elif '$var' in line:
                parts = line.split()
                var_idx = parts.index('$var')
                code = parts[var_idx + 3]
                name = parts[var_idx + 4]
                full_name = '.'.join(scope_stack + [name])
                var_map[code] = full_name
            elif '$enddefinitions' in line:
                in_defs = False
            i += 1

        # Parse value changes
        current_values = {name: 0 for name in var_map.values()}
        timestamps = []
        signal_data = {name: [] for name in var_map.values()}
        in_dumpvars = False

        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line == '$dumpvars':
                in_dumpvars = True
                i += 1
                continue
            if line == '$end':
                in_dumpvars = False
                i += 1
                continue
            if line.startswith('#'):
                ts = int(line[1:])
                timestamps.append(ts)
                # Snapshot current values
                i += 1
                # Read all changes at this timestamp
                while i < len(lines):
                    vline = lines[i].strip()
                    if not vline or vline.startswith('#') or vline.startswith('$'):
                        break
                    if vline.startswith('b'):
                        parts = vline.split()
                        bits = parts[0][1:]  # remove 'b'
                        code = parts[1]
                        if code in var_map:
                            current_values[var_map[code]] = int(bits, 2)
                    elif vline[0] in '01xXzZ':
                        code = vline[1:]
                        val = 1 if vline[0] == '1' else 0
                        if code in var_map:
                            current_values[var_map[code]] = val
                    i += 1
                for name in var_map.values():
                    signal_data[name].append(current_values[name])
                continue
            elif in_dumpvars and line.startswith('b'):
                parts = line.split()
                bits = parts[0][1:]
                code = parts[1]
                if code in var_map:
                    current_values[var_map[code]] = int(bits, 2)
            i += 1

        return cls(signal_data, timestamps)

    def signal(self, name: str, as_type=None) -> list:
        """Get signal values by hierarchical name."""
        raw = self._signals[name]
        if as_type is not None:
            return [as_type(v) for v in raw]
        return raw
```

**Step 5: Add compare method to Trace (in trace.py)**

```python
def compare(self, vcd, mapping: dict[str, str]) -> 'CompareResult':
    """Compare this trace against a VCD, signal by signal."""
    from rtlmod.io.vcd import CompareResult, CompareError
    mismatches = []
    for trace_name, vcd_name in mapping.items():
        vcd_vals = vcd.signal(vcd_name, as_type=self._types[trace_name])
        for cycle_idx, cycle_data in enumerate(self._cycles):
            if cycle_idx >= len(vcd_vals):
                break
            expected = cycle_data[trace_name]
            actual = vcd_vals[cycle_idx]
            if expected != actual:
                mismatches.append(CompareError(
                    cycle=cycle_idx,
                    signal=trace_name,
                    expected=expected,
                    actual=actual,
                ))
    return CompareResult(passed=len(mismatches) == 0, mismatches=mismatches)
```

**Step 6: Export VCD from rtlmod/io/__init__.py**

```python
from rtlmod.io.trace import Trace
from rtlmod.io.vcd import VCD
```

**Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_vcd.py -v`
Expected: All PASS

**Step 8: Commit**

```bash
git add rtlmod/io/vcd.py rtlmod/io/__init__.py rtlmod/io/trace.py tests/test_vcd.py tests/fixtures/
git commit -m "feat: VCD reader and trace comparison"
```

---

### Task 11: Pipeline Modeling

**Files:**
- Modify: `rtlmod/pipeline.py`
- Modify: `rtlmod/__init__.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write failing tests**

Create `tests/test_pipeline.py`:

```python
from dataclasses import dataclass
from rtlmod import UInt, SInt, Pipeline, PipelineSection
from rtlmod.io import Trace


u8 = UInt[8]
u16 = UInt[16]
u1 = UInt[1]


@dataclass
class SimpleState:
    value: u8 = u8(0)
    valid: u1 = u1(0)


def increment(s: SimpleState) -> SimpleState:
    return SimpleState(value=(s.value + u8(1)).resize(8), valid=s.valid)


class TestHomogeneousPipeline:
    def test_basic_flow(self):
        pipe = Pipeline(stage=increment, state_type=SimpleState, depth=3)
        pipe.inject(SimpleState(value=u8(10), valid=u1(1)))
        # Tick through 3 stages
        for _ in range(3):
            pipe.tick()
        outputs = list(pipe.outputs())
        assert len(outputs) == 1
        assert outputs[0].value == u8(13)  # 10 + 3 increments
        assert outputs[0].valid == u1(1)

    def test_empty_pipeline(self):
        pipe = Pipeline(stage=increment, state_type=SimpleState, depth=3)
        for _ in range(3):
            pipe.tick()
        assert list(pipe.outputs()) == []

    def test_multiple_inputs(self):
        pipe = Pipeline(stage=increment, state_type=SimpleState, depth=2)
        pipe.inject(SimpleState(value=u8(10), valid=u1(1)))
        pipe.tick()
        pipe.inject(SimpleState(value=u8(20), valid=u1(1)))
        pipe.tick()
        # First input exits after 2 ticks
        outputs = list(pipe.outputs())
        assert len(outputs) == 1
        assert outputs[0].value == u8(12)

        pipe.tick()
        outputs = list(pipe.outputs())
        assert len(outputs) == 1
        assert outputs[0].value == u8(22)


# Heterogeneous pipeline test
@dataclass
class WideState:
    value: u16 = u16(0)
    valid: u1 = u1(0)


def widen(s: SimpleState) -> WideState:
    return WideState(value=u16(s.value.value * 2), valid=s.valid)


def wide_increment(s: WideState) -> WideState:
    return WideState(value=(s.value + u16(1)).resize(16), valid=s.valid)


class TestHeterogeneousPipeline:
    def test_two_sections(self):
        pipe = Pipeline([
            PipelineSection(stage=increment, state_type=SimpleState, depth=2),
            PipelineSection(stage=widen, state_type=WideState, depth=1),
            PipelineSection(stage=wide_increment, state_type=WideState, depth=2),
        ])
        pipe.inject(SimpleState(value=u8(10), valid=u1(1)))
        # Total depth: 2 + 1 + 2 = 5
        for _ in range(5):
            pipe.tick()
        outputs = list(pipe.outputs())
        assert len(outputs) == 1
        # 10 + 2 increments = 12, widen *2 = 24, + 2 wide increments = 26
        assert outputs[0].value == u16(26)


class TestPipelineTrace:
    def test_trace(self):
        pipe = Pipeline(stage=increment, state_type=SimpleState, depth=3)
        pipe.inject(SimpleState(value=u8(5), valid=u1(1)))
        for _ in range(3):
            pipe.tick()
        trace = pipe.trace()
        assert isinstance(trace, Trace)
        assert len(trace) == 3
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL

**Step 3: Implement Pipeline in rtlmod/pipeline.py**

```python
"""Lightweight pipeline simulation helper."""

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Callable, Any


@dataclass
class PipelineSection:
    """One section of a pipeline with uniform stage logic."""
    stage: Callable
    state_type: type
    depth: int


class Pipeline:
    """Shift-register pipeline simulator.

    Usage (homogeneous):
        pipe = Pipeline(stage=fn, state_type=State, depth=N)

    Usage (heterogeneous):
        pipe = Pipeline([
            PipelineSection(stage=fn1, state_type=State1, depth=N1),
            PipelineSection(stage=fn2, state_type=State2, depth=N2),
        ])
    """

    def __init__(self, sections=None, *, stage=None, state_type=None, depth=None):
        if sections is not None:
            if isinstance(sections, list):
                self._sections = sections
            else:
                raise TypeError("sections must be a list of PipelineSection")
        elif stage is not None and state_type is not None and depth is not None:
            self._sections = [PipelineSection(stage=stage, state_type=state_type, depth=depth)]
        else:
            raise ValueError("Provide either sections list or stage+state_type+depth")

        # Build flat stage list
        self._stages: list[tuple[Callable, Any | None]] = []  # (stage_fn, state_or_None)
        for sec in self._sections:
            for _ in range(sec.depth):
                self._stages.append((sec.stage, None))

        self._pending_input = None
        self._output_buffer: list = []
        self._trace_data: list[dict] = []

    @property
    def depth(self) -> int:
        return len(self._stages)

    def inject(self, state):
        """Queue an input to enter the pipeline on next tick."""
        self._pending_input = state

    def tick(self):
        """Advance pipeline by one cycle."""
        self._output_buffer.clear()

        # Process from tail to head (so values shift forward)
        new_stages = list(self._stages)

        # Capture output (last stage)
        _, last_val = self._stages[-1]
        if last_val is not None:
            self._output_buffer.append(last_val)

        # Shift: each stage gets result of applying its function to previous stage's value
        for i in range(len(self._stages) - 1, 0, -1):
            fn_i, _ = self._stages[i]
            _, prev_val = self._stages[i - 1]
            if prev_val is not None:
                new_stages[i] = (fn_i, fn_i(prev_val))
            else:
                new_stages[i] = (fn_i, None)

        # First stage gets input
        fn_0, _ = self._stages[0]
        if self._pending_input is not None:
            new_stages[0] = (fn_0, fn_0(self._pending_input))
            self._pending_input = None
        else:
            new_stages[0] = (fn_0, None)

        self._stages = new_stages

        # Record trace data
        cycle_data = {}
        for i, (_, val) in enumerate(self._stages):
            if val is not None:
                for f in fields(val):
                    cycle_data[f"stage{i}_{f.name}"] = getattr(val, f.name)
        if cycle_data:
            self._trace_data.append(cycle_data)

    def outputs(self):
        """Iterate over values that exited the pipeline on the last tick."""
        return iter(self._output_buffer)

    def trace(self):
        """Return a Trace object with all recorded pipeline state."""
        from rtlmod.io.trace import Trace
        trace = Trace()
        for cycle_data in self._trace_data:
            trace.record(**cycle_data)
        return trace
```

**Step 4: Export from rtlmod/__init__.py**

```python
from rtlmod.types import UInt, SInt
from rtlmod.ops import concat
from rtlmod.pipeline import Pipeline, PipelineSection
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add rtlmod/pipeline.py rtlmod/__init__.py tests/test_pipeline.py
git commit -m "feat: Pipeline with homogeneous and heterogeneous sections"
```

---

### Task 12: Integration Test - Full Workflow

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write an end-to-end test**

Create `tests/test_integration.py` that exercises the full workflow as a user would: define types, build a simple model, run it, trace it, export to CSV and .mem, and verify output.

```python
"""Integration test: model a simple 4-stage accumulator pipeline."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from rtlmod import UInt, SInt, SInt, concat, Pipeline, PipelineSection
from rtlmod.io import Trace

u16 = UInt[16]
s16 = SInt[16]
u1 = UInt[1]


@dataclass
class AccState:
    acc: u16 = u16(0)
    addend: u16 = u16(0)
    valid: u1 = u1(0)


def acc_stage(s: AccState) -> AccState:
    new_acc = (s.acc + s.addend).resize(16)
    return AccState(acc=new_acc, addend=s.addend, valid=s.valid)


class TestFullWorkflow:
    def test_model_trace_export(self):
        # 1. Build pipeline
        pipe = Pipeline(stage=acc_stage, state_type=AccState, depth=4)

        # 2. Run model
        pipe.inject(AccState(acc=u16(0), addend=u16(10), valid=u1(1)))
        trace = Trace()
        for cycle in range(6):
            pipe.tick()
            for out in pipe.outputs():
                trace.record(acc=out.acc, valid=out.valid)

        # 3. Verify output: 0 + 10*4 = 40
        assert len(trace) == 1
        assert trace[0].acc == u16(40)

        # 4. Export to CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "acc_test.csv"
            trace.to_csv(str(csv_path))
            assert csv_path.exists()

            # 5. Export to .mem
            trace.to_mem(tmpdir)
            assert (Path(tmpdir) / "acc.mem").exists()
            assert (Path(tmpdir) / "valid.mem").exists()

            # 6. Reload CSV and verify
            loaded = Trace.from_csv(str(csv_path), types={'acc': u16, 'valid': u1})
            assert loaded[0].acc == u16(40)

    def test_bit_operations_workflow(self):
        """Test typical RTL modeling patterns."""
        s258 = SInt[258]
        u256 = UInt[256]

        # Wide arithmetic
        a = s258(1000000)
        b = s258(-500000)
        c = a + b
        assert c.value == 500000
        assert c.width == 259

        # Resize back
        d = c.resize(258)
        assert d.width == 258
        assert d.value == 500000

        # Bit slicing
        x = u256(0xDEADBEEF)
        assert x[7:0].value == 0xEF
        assert x[15:8].value == 0xBE

        # Concat
        hi = UInt[8](0xAB)
        lo = UInt[8](0xCD)
        packed = concat(hi, lo)
        assert packed.value == 0xABCD
        assert packed.width == 16
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/test_integration.py -v`
Expected: All PASS

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full workflow"
```

---

### Task 13: Final Cleanup and Full Test Run

**Files:**
- Review all files for consistency
- Verify all `__init__.py` exports are correct

**Step 1: Run full test suite with coverage**

Run:
```bash
uv pip install pytest-cov
uv run pytest tests/ -v --cov=rtlmod --cov-report=term-missing
```

Expected: All tests pass, reasonable coverage.

**Step 2: Verify package installs cleanly**

Run:
```bash
uv pip install -e . --force-reinstall
uv run python -c "from rtlmod import SInt, UInt, concat, Pipeline, PipelineSection; from rtlmod.io import Trace, VCD; print('All imports OK')"
```

Expected: "All imports OK"

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: final cleanup and export verification"
```
