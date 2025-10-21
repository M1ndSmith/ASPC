"""
MSA System
Measurement System Analysis (MSA) with AI Agent capabilities

This package contains:
- MSAPipeline: Core MSA analysis engine
- MSA Tool: Single comprehensive LangGraph tool
- MSA Agent: Conversational AI interface

Supported Studies:
- Gage R&R (Repeatability & Reproducibility)
- Bias Analysis
- Linearity Studies
- Stability Analysis
"""

from .msa_pipeline import MSAPipeline
from .msa_tools import run_msa_analysis

__version__ = "2.0.0"
__author__ = "MSA AI Team"

__all__ = [
    "MSAPipeline",
    "run_msa_analysis"
]
