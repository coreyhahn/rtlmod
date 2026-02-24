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
            return "PASS: all cycles matched"
        return (f"FAIL: {len(self.mismatches)} mismatch(es). "
                f"First at cycle {self.first_error.cycle}, "
                f"signal '{self.first_error.signal}': "
                f"expected {self.first_error.expected}, "
                f"got {self.first_error.actual}")


class VCD:
    """VCD file reader."""

    def __init__(self, signals: dict[str, list], timestamps: list[int]):
        self._signals = signals
        self._timestamps = timestamps

    @classmethod
    def read(cls, path: str | Path) -> VCD:
        path = Path(path)
        text = path.read_text()

        # Parse header
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
                i += 1
                while i < len(lines):
                    vline = lines[i].strip()
                    if not vline or vline.startswith('#') or vline.startswith('$'):
                        break
                    if vline.startswith('b'):
                        parts = vline.split()
                        bits = parts[0][1:]
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
        raw = self._signals[name]
        if as_type is not None:
            return [as_type(v) for v in raw]
        return raw
