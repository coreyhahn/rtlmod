# rtlmod - Python RTL Modeling Library Design

## Problem

Python models of RTL/Verilog designs require repetitive boilerplate: manual bit masking, `to_signed()`/`to_unsigned()` helpers reimplemented per project, ad-hoc test vector export, and custom pipeline simulation code. This wastes time for both human and AI authors.

## Solution

A single library (`rtlmod`) providing numpy-like arbitrary-width integer types that enforce bit-accurate Verilog semantics automatically. Two layers:

- **Core types** (`rtlmod.types`): `SInt[N]`, `UInt[N]` with operator overloading
- **IO layer** (`rtlmod.io`): Trace recording, CSV/mem export, VCD comparison

## Core Types

### Type Declaration

```python
from rtlmod import SInt, UInt, concat

s258 = SInt[258]   # reusable signed 258-bit type
u13  = UInt[13]    # reusable unsigned 13-bit type

f = s258(0)        # scalar
img = u8.array(shape=(480, 640))  # numpy-backed array
```

### Arithmetic Rules (Verilog Semantics)

- Addition: `SInt[N] + SInt[N]` -> `SInt[N+1]`
- Multiplication: `SInt[N] * SInt[N]` -> `SInt[2N]`
- Shifts: `SInt[N] << k` -> `SInt[N]` (width preserved)
- Mixed signedness: unsigned promotes to signed, width = max + 1
- All operations eager-mask to declared width. No manual masking.

### Resizing

```python
result = (a + b).resize(13)                    # truncate (default)
result = (a + b).resize(13, round='saturate')  # clamp to min/max
val.sign_extend(32)                            # -> SInt[32]
```

### Bit Operations

```python
val[7:0]           # bit slice -> UInt[8], Verilog-style MSB:LSB
val[31]            # single bit -> UInt[1]
concat(a, b, c)    # width = sum of input widths
val.xor_reduce()   # -> UInt[1]
val.and_reduce()   # -> UInt[1]
val.or_reduce()    # -> UInt[1]
```

### Display

```python
print(f)     # s258'd0
print(f.hex) # 258'sh0000...0
print(f.bin) # 258'sb0000...0
repr(f)      # SInt[258](0)
```

## Array Operations

Numpy-backed, element-wise bit-accurate arithmetic:

```python
s16 = SInt[16]
u8  = UInt[8]

pixels = u8.array(shape=(480, 640))
noise  = u8.random(shape=(480, 640))
ramp   = u8.arange(0, 256)
coeffs = s16.array([100, -200, 300, -400])

# Arithmetic follows same width rules as scalars
result = pixels + pixels    # UInt[9] array
squared = coeffs * coeffs   # SInt[32] array

# Numpy interop
raw_np = pixels.to_numpy()
back   = u8.from_numpy(raw_np)
```

All operations are eager (mask after every op).

## IO Layer

### Trace Recording and CSV Export

```python
from rtlmod.io import Trace

trace = Trace()
for cycle in range(1000):
    state = model.tick()
    trace.record(f=state.f, g=state.g, delta=state.delta, valid=state.valid)

trace.to_csv("test_vectors.csv")                  # hex by default
trace.to_csv("test_vectors.csv", fmt='decimal')
trace.to_csv("test_vectors.csv", fmt='binary')
```

### .mem File Export

```python
trace.to_mem("output_dir/")                # one file per signal
trace.to_mem("f_only.mem", signals=['f'])   # single signal
```

### VCD Reading and RTL Comparison

```python
from rtlmod.io import VCD

vcd = VCD.read("sim_output.vcd")
rtl_f = vcd.signal("tb.dut.stage[0].f", as_type=s258)

diff = trace.compare(vcd, mapping={
    'f': 'tb.dut.stage[0].f',
    'g': 'tb.dut.stage[0].g',
})

# diff.passed, diff.first_error, diff.summary(), diff.mismatches
```

### Loading Traces

```python
trace = Trace.from_csv("test_vectors.csv", types={
    'f': s258, 'g': s258, 'delta': SInt[11], 'valid': UInt[1]
})
for cycle in trace:
    print(cycle.f, cycle.g)
```

## Pipeline Modeling

### Simple Homogeneous Pipeline

```python
from rtlmod import Pipeline, PipelineSection

pipe = Pipeline(stage=divstep, state_type=State, depth=590)
pipe.inject(initial_state)
for _ in range(600):
    pipe.tick()
for result in pipe.outputs():
    print(result.f)
```

### Heterogeneous Pipeline (Multiple Sections)

```python
pipe = Pipeline([
    PipelineSection(stage=phase1_divstep, state_type=Phase1State, depth=380),
    PipelineSection(stage=width_transition, state_type=Phase2State, depth=1),
    PipelineSection(stage=phase2_divstep, state_type=Phase2State, depth=210),
])
```

### Pipeline Tracing

```python
trace = pipe.trace()
trace.to_csv("pipeline_dump.csv")
```

## Package Structure

```
rtlmod/
├── __init__.py      # exports: SInt, UInt, concat, Pipeline, PipelineSection
├── types.py         # SInt[N], UInt[N] scalar implementation
├── array.py         # numpy-backed array implementation
├── ops.py           # concat, bit slice, reductions
├── pipeline.py      # Pipeline, PipelineSection
└── io/
    ├── __init__.py  # exports: Trace, VCD
    ├── trace.py     # Trace recorder, CSV/mem export and import
    └── vcd.py       # VCD reader and comparison
```

## Dependencies

- `numpy` (required) - array backing store
- No other runtime dependencies

## Design Decisions

1. **Eager masking** - every operation masks immediately. No lazy evaluation. Correctness over performance.
2. **Verilog width rules** - addition grows by 1, multiplication doubles, shifts preserve. Matches what the RTL will do.
3. **Type factory pattern** - `SInt[258]` creates a reusable type object. Avoids repeating width on every construction.
4. **Numpy backing for arrays** - uses next-larger native dtype internally, masks after every operation. Gets numpy C speed.
5. **Trace as central IO object** - records named signals per cycle, exports to CSV (primary), .mem, and compares against VCD.
6. **Minimal pipeline helper** - shift register of states with per-stage function. Supports heterogeneous sections via PipelineSection list. Not a full simulation framework.

## Audience

Both human authors and AI (Claude) generating models. The API is explicit enough for AI to produce correct code (clear width declarations, no magic) and concise enough for hand-writing.
