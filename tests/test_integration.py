"""Integration test: exercises the full rtlmod workflow."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from rtlmod import UInt, SInt, concat, Pipeline, PipelineSection
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

    def test_array_image_processing(self):
        """Simulate a simple image processing pipeline."""
        u8 = UInt[8]
        s17 = SInt[17]

        # Create a small "image"
        pixels = u8.array([100, 150, 200, 50, 75, 25])

        # Roberts-like operation: difference of adjacent pixels
        left = pixels[0:5]   # this is numpy-style slicing, not bit slicing
        right = pixels[1:6]

        # Subtraction always produces signed result
        diff = left - right
        assert diff.signed == True
        assert diff.width == 9  # max(8,8) + 1

        # Verify specific values
        assert diff[0].value == -50   # 100 - 150
        assert diff[1].value == -50   # 150 - 200
        assert diff[2].value == 150   # 200 - 50
