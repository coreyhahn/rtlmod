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
