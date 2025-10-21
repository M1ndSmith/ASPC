"""
Process Capability System
Process Capability Analysis with AI Agent capabilities

This package contains:
- ProcessCapabilityPipeline: Core capability analysis engine
- Capability Tool: Single comprehensive LangGraph tool
- Capability Agent: Conversational AI interface

Supported Analysis:
- Cp, Cpk (Short-term capability)
- Pp, Ppk (Long-term performance)
- Yield and DPMO analysis
- Process centering assessment
"""

from .capability_pipeline import ProcessCapabilityPipeline
from .capability_tools import run_capability_analysis

__version__ = "2.0.0"
__author__ = "Process Capability AI Team"

__all__ = [
    "ProcessCapabilityPipeline",
    "run_capability_analysis"
]
