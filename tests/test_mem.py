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
            assert content == 'ffff'
