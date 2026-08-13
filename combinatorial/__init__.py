"""Combinatorial dual-mode test package for spc_core."""
from __future__ import annotations

from combinatorial.matrix import CaseSpec, build_matrix, load_config
from combinatorial.schema_catalog import SchemaCatalog, load_catalog

__all__ = [
    "CaseSpec",
    "SchemaCatalog",
    "build_matrix",
    "load_catalog",
    "load_config",
]
