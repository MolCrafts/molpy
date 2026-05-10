"""Stage-level reporting for 3D generation.

Mirrors the molrs ``EmbedReport`` and ``StageReport`` types in plain Python
dataclasses so molpy callers do not have to keep a live reference to the
underlying Rust object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import molrs


@dataclass(frozen=True)
class StageReport:
    """Per-stage execution metrics."""

    stage: str
    energy_before: float | None
    energy_after: float | None
    steps: int
    converged: bool
    elapsed_ms: int

    @classmethod
    def from_native(cls, native: "molrs.StageReport") -> "StageReport":
        return cls(
            stage=str(native.stage),
            energy_before=native.energy_before,
            energy_after=native.energy_after,
            steps=int(native.steps),
            converged=bool(native.converged),
            elapsed_ms=int(native.elapsed_ms),
        )


@dataclass(frozen=True)
class EmbedReport:
    """End-to-end generation report."""

    final_energy: float | None
    warnings: list[str] = field(default_factory=list)
    stages: list[StageReport] = field(default_factory=list)

    @classmethod
    def from_native(cls, native: "molrs.EmbedReport") -> "EmbedReport":
        return cls(
            final_energy=native.final_energy,
            warnings=list(native.warnings),
            stages=[StageReport.from_native(s) for s in native.stages],
        )
