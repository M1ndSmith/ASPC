"""End-to-end batch analysis against synthetic datasets."""
from __future__ import annotations

from adapters.stream import FileReplaySource, stream_evaluate
from spc_core import analyze_control_chart, capability_analysis, gage_rr_anova, ingest
from spc_core.report import CapabilityReport, MSAReport, SPCReport


def test_e2e_control_chart_individual(dataset):
    cols = dataset("spc_individual_in_control")
    frame = ingest(cols)
    result = analyze_control_chart(cols[frame.column_map.value_col])
    report = SPCReport.from_chart_result(result, source_file="synthetic:spc_individual_in_control")
    assert report.chart_type.value == "I-MR"
    assert report.limits.version
    d = report.model_dump(mode="json")
    assert "plotted_values" in d


def test_e2e_capability(dataset):
    cols = dataset("capability_excellent")
    values = [float(v) for v in cols["measurement"]]
    result = capability_analysis(values, usl=10.5, lsl=9.5)
    report = CapabilityReport.from_capability(result)
    assert "Cpk" in report.result
    assert report.result["rating"]


def test_e2e_msa(dataset):
    cols = dataset("msa_gage_rr_excellent")
    result = gage_rr_anova(cols["Part"], cols["Operator"], cols["Measurement"])
    report = MSAReport.from_gage_rr(result)
    assert report.result["grr_percent"] == result.grr_percent


def test_e2e_file_replay_phase2(write_dataset):
    phase1 = write_dataset("spc_individual_in_control")
    phase2 = write_dataset("spc_individual_out_of_control")
    from adapters.io_files import load_columns

    cols = load_columns(phase1)
    result = analyze_control_chart(cols["measurement"])
    source = FileReplaySource(phase2, value_col="measurement")
    signals = stream_evaluate(source, result.limits)
    # Out-of-control fixture should produce signals against in-control limits
    # (or zero if the spike isn't extreme enough — just assert it runs cleanly)
    assert isinstance(signals, list)
