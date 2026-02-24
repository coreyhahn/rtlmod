"""Lightweight pipeline simulation helper."""

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Callable


@dataclass
class PipelineSection:
    stage: Callable
    state_type: type
    depth: int


class Pipeline:
    def __init__(self, sections=None, *, stage=None, state_type=None, depth=None):
        if sections is not None and isinstance(sections, list):
            self._sections = sections
        elif stage is not None and state_type is not None and depth is not None:
            self._sections = [PipelineSection(stage=stage, state_type=state_type, depth=depth)]
        else:
            raise ValueError("Provide either sections list or stage+state_type+depth")

        # Build flat list of (stage_fn, current_value_or_None)
        self._stages: list[tuple[Callable, object | None]] = []
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
        self._pending_input = state

    def tick(self):
        self._output_buffer.clear()

        # Shift from tail to head: each stage applies its function to
        # the previous stage's value, propagating data forward one step.
        new_stages = list(self._stages)
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

        # Capture output from last stage (after the shift)
        _, last_val = self._stages[-1]
        if last_val is not None:
            self._output_buffer.append(last_val)

        # Record trace
        cycle_data = {}
        for i, (_, val) in enumerate(self._stages):
            if val is not None:
                for f in fields(val):
                    cycle_data[f"stage{i}_{f.name}"] = getattr(val, f.name)
        if cycle_data:
            self._trace_data.append(cycle_data)

    def outputs(self):
        return iter(self._output_buffer)

    def trace(self):
        from rtlmod.io.trace import Trace
        trace = Trace()
        for cycle_data in self._trace_data:
            trace.record_dynamic(**cycle_data)
        return trace
