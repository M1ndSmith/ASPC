"""Typed contracts for the SPC core.

These models are the stable, serializable boundary of the library. Everything that
crosses into an adapter (persistence, API, rendering) or is returned to a caller uses
these types. The core computation modules never touch I/O; they only produce/consume
these models and plain numpy arrays.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ChartType(str, Enum):
    """Supported control chart types."""

    I_MR = "I-MR"
    XBAR_R = "Xbar-R"
    XBAR_S = "Xbar-S"
    P = "P"
    NP = "NP"
    C = "C"
    U = "U"
    EWMA = "EWMA"
    CUSUM = "CUSUM"


class DataType(str, Enum):
    CONTINUOUS = "continuous"
    ATTRIBUTE = "attribute"


class Phase(str, Enum):
    """Phase I establishes and freezes limits; Phase II applies them to new data."""

    PHASE_I = "PHASE_I"
    PHASE_II = "PHASE_II"


class QualityFlag(str, Enum):
    """Provenance of a single measurement (SPC never silently imputes/drops)."""

    ORIGINAL = "ORIGINAL"
    IMPUTED_LOCF = "IMPUTED_LOCF"
    MISSING_SENSOR = "MISSING_SENSOR"
    EXCLUDED_MAINTENANCE = "EXCLUDED_MAINTENANCE"
    EXCLUDED_INCOMPLETE = "EXCLUDED_INCOMPLETE"
    RESTORED_FROM_BACKUP = "RESTORED_FROM_BACKUP"
    MISSING_HUMAN = "MISSING_HUMAN"


class DistributionFlag(str, Enum):
    NORMAL = "NORMAL"
    TRANSFORMED = "TRANSFORMED"
    NON_NORMAL_RAW = "NON_NORMAL_RAW"


class LimitSet(BaseModel):
    """Center line and control limits for one chart component (e.g. the X or the R panel).

    UCL/LCL may be a scalar (fixed limits) or a per-point list (variable limits for
    P and U charts where the subgroup size changes point to point).
    """

    model_config = ConfigDict(frozen=True)

    center: float
    ucl: Union[float, list[float]]
    lcl: Union[float, list[float]]

    def ucl_at(self, i: int) -> float:
        return self.ucl[i] if isinstance(self.ucl, list) else self.ucl

    def lcl_at(self, i: int) -> float:
        return self.lcl[i] if isinstance(self.lcl, list) else self.lcl


class ControlLimits(BaseModel):
    """Immutable, versioned Phase I limits.

    Frozen after computation. Phase II must consume these without recomputing them.
    `version` is a content hash so any downstream record can prove which limits it used.
    """

    model_config = ConfigDict(frozen=True)

    chart_type: ChartType
    subgroup_size: int
    # Component name -> LimitSet. e.g. {"individuals": ..., "moving_range": ...}
    components: dict[str, LimitSet]
    # Estimated within/short-term sigma of the plotted statistic (used by the rule engine).
    sigma: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_n_points: Optional[int] = None
    notes: dict[str, float] = Field(default_factory=dict)

    @property
    def primary(self) -> LimitSet:
        """The component the run-rules are evaluated against (first declared component)."""
        return next(iter(self.components.values()))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def version(self) -> str:
        """Content hash so any downstream record can prove which limits it used.

        Serialized by ``model_dump()`` via pydantic ``computed_field``.
        """
        payload = {
            "chart_type": self.chart_type.value,
            "subgroup_size": self.subgroup_size,
            "components": {
                name: {"center": ls.center, "ucl": ls.ucl, "lcl": ls.lcl}
                for name, ls in self.components.items()
            },
            "sigma": self.sigma,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


class Signal(BaseModel):
    """A detected out-of-control condition."""

    rule_id: str
    rule_name: str
    index: int
    value: float
    description: str
    side: Optional[str] = None  # "upper" | "lower" | None

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"[{self.rule_id}] point {self.index}: {self.description}"


class SPCRecord(BaseModel):
    """The SPC-ready output schema (one plotted point).

    Mirrors the minimum-fields schema from the MVP design document so every point
    carries its provenance, distribution handling, phase, and the limits it was judged
    against.
    """

    timestamp: Optional[datetime] = None
    subgroup_id: Optional[int] = None
    measurement_value: float
    data_quality_flag: QualityFlag = QualityFlag.ORIGINAL
    distribution_flag: DistributionFlag = DistributionFlag.NORMAL
    transform_applied: Optional[str] = None
    phase: Phase = Phase.PHASE_I
    ucl: Optional[float] = None
    lcl: Optional[float] = None
    centerline: Optional[float] = None
    gage_id: Optional[str] = None
    machine_id: Optional[str] = None
    signals: list[Signal] = Field(default_factory=list)
