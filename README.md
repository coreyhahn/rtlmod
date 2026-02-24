# rtlmod

[![CI](https://github.com/coreyhahn/rtlmod/actions/workflows/ci.yml/badge.svg)](https://github.com/coreyhahn/rtlmod/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Numpy-like arbitrary-width integer types for Python RTL modeling. Write bit-accurate Verilog reference models without manual masking.

## Install

```bash
pip install rtlmod
```

## Quick Start

```python
from rtlmod import SInt, UInt, concat

# Define types once, reuse everywhere
s16 = SInt[16]
u8  = UInt[8]

# Arithmetic follows Verilog width rules automatically
a = s16(1000)
b = s16(-500)
c = a + b           # SInt[17], no overflow, no manual masking
d = (a * b).resize(16)  # truncate back to 16-bit

# Bit operations
lsb = a[0]          # UInt[1]
low_byte = a[7:0]   # UInt[8]
packed = concat(a, b)  # SInt[32]
parity = a.xor_reduce()  # UInt[1]
```

## Why

Python models of RTL designs are full of repetitive boilerplate:

```python
# Before: manual masking everywhere
result = (a + b) & ((1 << 16) - 1)
if result >= (1 << 15):
    result -= (1 << 16)
```

```python
# After: just math
result = a + b  # width, signedness, and masking handled automatically
```

rtlmod gives you:
- **Arbitrary-width integers** (13-bit, 258-bit, whatever your RTL uses)
- **Verilog arithmetic semantics** (width growth on add/multiply, width preservation on shift)
- **Numpy-backed arrays** for image processing and bulk operations
- **Test vector export** to CSV and `.mem` files
- **VCD comparison** to diff Python models against RTL simulation output

## Core Types

```python
from rtlmod import SInt, UInt

# Type factory - creates reusable types like numpy.int32
s258 = SInt[258]    # signed 258-bit
u13  = UInt[13]     # unsigned 13-bit

# Scalars
f = s258(0)
g = u13(42)

# Arrays (numpy-backed)
img = UInt[8].array(shape=(480, 640))
coeffs = SInt[16].array([100, -200, 300, -400])
```

### Arithmetic Rules

| Operation | Result Width | Example |
|-----------|-------------|---------|
| `a + b` | `max(wa, wb) + 1` | `SInt[16] + SInt[16]` -> `SInt[17]` |
| `a * b` | `wa + wb` | `SInt[16] * SInt[16]` -> `SInt[32]` |
| `a << n` | `wa` (preserved) | `SInt[16] << 3` -> `SInt[16]` |
| `a >> n` | `wa` (preserved) | `SInt[16] >> 3` -> `SInt[16]` |
| unsigned + signed | signed, `max + 1` | `UInt[13] + SInt[16]` -> `SInt[17]` |

### Resizing

```python
narrow = wide_val.resize(16)                     # truncate (default)
clamped = wide_val.resize(16, round='saturate')  # clamp to min/max
extended = narrow_val.sign_extend(32)             # sign-extend to 32-bit
```

### Display

```python
x = SInt[16](-1)
print(x)       # s16'd-1
print(x.hex)   # 16'shffff
print(x.bin)   # 16'sb1111111111111111
repr(x)        # SInt[16](-1)
```

## Arrays

Numpy performance with bit-accurate semantics:

```python
u8 = UInt[8]

pixels = u8.array(shape=(480, 640))
noise  = u8.random(shape=(480, 640))
ramp   = u8.arange(0, 256)

# Element-wise arithmetic, auto-masked
bright = pixels + u8(50)   # UInt[9], no overflow wrapping bugs

# Numpy interop
raw = pixels.to_numpy()
back = u8.from_numpy(raw)
```

## Test Vector IO

### Recording and Export

```python
from rtlmod.io import Trace

trace = Trace()
for cycle in range(1000):
    state = model.tick()
    trace.record(f=state.f, g=state.g, valid=state.valid)

# CSV (columnar, primary format)
trace.to_csv("vectors.csv")
trace.to_csv("vectors.csv", fmt='decimal')

# .mem files (Verilog $readmemh compatible)
trace.to_mem("output_dir/")           # one file per signal
trace.to_mem("f.mem", signals=['f'])   # single signal
```

### RTL Comparison

```python
from rtlmod.io import VCD

vcd = VCD.read("sim_output.vcd")
diff = trace.compare(vcd, mapping={
    'f': 'tb.dut.f',
    'g': 'tb.dut.g',
})

print(diff.summary())
# PASS: 1000 cycles, 2 signals, 0 mismatches
```

### Loading Traces

```python
trace = Trace.from_csv("vectors.csv", types={
    'f': SInt[258], 'g': SInt[258], 'valid': UInt[1]
})
for cycle in trace:
    print(cycle.f, cycle.g)
```

## Pipeline Modeling

```python
from rtlmod import Pipeline, PipelineSection
from dataclasses import dataclass

@dataclass
class State:
    f: SInt[258] = SInt[258](0)
    g: SInt[258] = SInt[258](0)
    delta: SInt[11] = SInt[11](1)

def divstep(s: State) -> State:
    # your stage logic
    ...

# Homogeneous pipeline
pipe = Pipeline(stage=divstep, state_type=State, depth=590)

# Heterogeneous pipeline (different widths/logic per section)
pipe = Pipeline([
    PipelineSection(stage=phase1_step, state_type=Phase1State, depth=380),
    PipelineSection(stage=transition,  state_type=Phase2State, depth=1),
    PipelineSection(stage=phase2_step, state_type=Phase2State, depth=210),
])

pipe.inject(initial_state)
for _ in range(600):
    pipe.tick()

for result in pipe.outputs():
    print(result)

# Export all pipeline state
pipe.trace().to_csv("pipeline.csv")
```

## License

MIT
