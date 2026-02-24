from pathlib import Path
from rtlmod import UInt
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
        trace.record(x=u8(99))

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
