"""Trace recorder for cycle-by-cycle signal capture."""

from __future__ import annotations
import csv
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
        """Record one cycle of signal values.

        On first call, establishes the signal names and types.
        Subsequent calls must provide the same signals (raises ValueError if not).
        Use record_dynamic() for traces with varying signal sets (e.g., pipeline traces).
        """
        if not self._signals:
            self._signals = list(kwargs.keys())
            self._types = {name: type(val) for name, val in kwargs.items()}
        else:
            if list(kwargs.keys()) != self._signals:
                raise ValueError(
                    f"Signal mismatch: expected {self._signals}, got {list(kwargs.keys())}"
                )
        self._cycles.append(dict(kwargs))

    def record_dynamic(self, **kwargs):
        """Record one cycle, updating signal list to include any new signals."""
        if not self._signals:
            self._signals = list(kwargs.keys())
            self._types = {name: type(val) for name, val in kwargs.items()}
        else:
            for name, val in kwargs.items():
                if name not in self._types:
                    self._signals.append(name)
                    self._types[name] = type(val)
        self._cycles.append(dict(kwargs))

    def __getitem__(self, index):
        return SimpleNamespace(**self._cycles[index])

    def __iter__(self):
        for cycle_data in self._cycles:
            yield SimpleNamespace(**cycle_data)

    def _format_value(self, val, fmt: str) -> str:
        """Format a signal value as hex, decimal, or binary string."""
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
        """Export trace to CSV file.

        Args:
            path: Output file path.
            fmt: Value format - 'hex' (default), 'decimal', or 'binary'.
        """
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
    def from_csv(cls, path: str, types: dict[str, type], fmt: str = 'hex') -> Trace:
        """Load a trace from a CSV file.

        Args:
            path: Input CSV file path.
            types: Mapping of signal name to type (e.g. {'a': UInt[8], 'b': SInt[16]}).
            fmt: Value format used in the CSV - 'hex' (default), 'decimal', or 'binary'.

        Returns:
            A new Trace with the loaded data.
        """
        trace = Trace()
        p = Path(path)
        with open(p, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kwargs = {}
                for name, typ in types.items():
                    raw = row[name].strip()
                    if fmt == 'hex':
                        val = int(raw, 16)
                    elif fmt == 'binary':
                        val = int(raw, 2)
                    else:
                        val = int(raw)
                    kwargs[name] = typ(val)
                trace.record(**kwargs)
        return trace

    def to_mem(self, path: str, signals: list[str] | None = None):
        """Export trace signals to .mem files (Verilog $readmemh format).

        Args:
            path: Directory for per-signal .mem files, or a single file path.
            signals: Subset of signals to export. Defaults to all signals.
        """
        p = Path(path)
        sigs = signals or self._signals
        if p.is_dir() or (not p.suffix and not p.exists()):
            # Directory mode: one .mem file per signal
            p.mkdir(parents=True, exist_ok=True)
            for sig in sigs:
                self._write_mem_file(p / f"{sig}.mem", sig)
        else:
            # Single file mode: requires exactly one signal
            if len(sigs) == 1:
                self._write_mem_file(p, sigs[0])
            else:
                raise ValueError("Single file export requires exactly one signal")

    def _write_mem_file(self, path: Path, signal: str):
        """Write a single .mem file with one hex value per line."""
        with open(path, 'w') as f:
            for cycle_data in self._cycles:
                val = cycle_data[signal]
                raw = val._to_unsigned()
                hex_digits = (val.width + 3) // 4
                f.write(f"{raw:0{hex_digits}x}\n")

    def compare(self, vcd, mapping: dict[str, str]):
        """Compare this trace against a VCD, signal by signal.

        Args:
            vcd: A VCD reader object.
            mapping: Mapping of trace signal names to VCD signal names.

        Returns:
            A CompareResult with pass/fail status and mismatches.
        """
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
                        cycle=cycle_idx, signal=trace_name,
                        expected=expected, actual=actual,
                    ))
        return CompareResult(passed=len(mismatches) == 0, mismatches=mismatches)
