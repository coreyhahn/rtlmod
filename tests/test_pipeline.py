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
        for _ in range(3):
            pipe.tick()
        outputs = list(pipe.outputs())
        assert len(outputs) == 1
        assert outputs[0].value == u8(13)
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
        outputs = list(pipe.outputs())
        assert len(outputs) == 1
        assert outputs[0].value == u8(12)

        pipe.tick()
        outputs = list(pipe.outputs())
        assert len(outputs) == 1
        assert outputs[0].value == u8(22)


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
