"""
Control Chart System
Statistical Process Control (SPC) AI Agent with LangGraph

This package contains:
- ControlChartPipeline: Core SPC analysis engine
- Control Chart Tool: Single comprehensive LangGraph tool
- Control Chart Agent: Conversational AI interface

Supported Charts:
- Continuous: I-MR, Xbar-R, Xbar-S
- Attribute: P, NP, C, U
"""

from .control_chart_pipeline import ControlChartPipeline
from .control_chart_tools import run_control_chart_analysis

__version__ = "2.0.0"
__author__ = "Control Chart AI Team"

__all__ = [
    "ControlChartPipeline",
    "run_control_chart_analysis"
]
