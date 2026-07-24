"""ASPC CLI — thin consumer of spc_core (no LLM)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adapters.io_files import FileReadError, load_columns
from adapters.persistence import SQLiteRepository
from adapters.render_plotly import (
    render_capability_html,
    render_control_chart_html,
    render_msa_html,
    save_html,
)
from apps.config import get_config
from spc_core import (
    ChartType,
    analyze_control_chart,
    bias_study,
    capability_analysis,
    check_autocorrelation,
    check_normality,
    gage_rr_anova,
    gage_rr_range,
    ingest,
    linearity_study,
    stability_study,
)
from spc_core.report import CapabilityReport, MSAReport, SPCReport


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aspc",
        description="ASPC — Statistical Process Control (core library CLI)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # control-chart
    cc = sub.add_parser("control-chart", help="Run control chart analysis")
    cc.add_argument("-f", "--file", required=True)
    cc.add_argument("--value-col")
    cc.add_argument("--subgroup-col")
    cc.add_argument("--sample-size-col")
    cc.add_argument("--opportunity-col")
    cc.add_argument("--chart-type", choices=[c.value for c in ChartType])
    cc.add_argument("--ruleset", default=None)
    cc.add_argument("--html", help="Write HTML report to this path")
    cc.add_argument("--json", action="store_true", help="Print JSON report")

    # capability
    cap = sub.add_parser("capability", help="Run process capability analysis")
    cap.add_argument("-f", "--file", required=True)
    cap.add_argument("--usl", type=float, required=True)
    cap.add_argument("--lsl", type=float, required=True)
    cap.add_argument("--target", type=float)
    cap.add_argument("--value-col")
    cap.add_argument("--subgroup-col")
    cap.add_argument("--html")
    cap.add_argument("--json", action="store_true")

    # msa
    msa = sub.add_parser("msa", help="Run MSA / Gage R&R / bias / linearity / stability")
    msa.add_argument("-f", "--file", required=True)
    msa.add_argument("--study-type", choices=["gage_rr", "bias", "linearity", "stability"])
    msa.add_argument("--method", choices=["anova", "range"], default="anova")
    msa.add_argument("--tolerance", type=float)
    msa.add_argument("--part-col")
    msa.add_argument("--operator-col")
    msa.add_argument("--measurement-col")
    msa.add_argument("--reference-col")
    msa.add_argument("--html")
    msa.add_argument("--json", action="store_true")

    # serve
    srv = sub.add_parser("serve", help="Start the FastAPI server")
    srv.add_argument("--host", default=None)
    srv.add_argument("--port", type=int, default=None)

    args = parser.parse_args(argv)
    cfg = get_config()

    if args.cmd == "serve":
        import uvicorn
        from apps.api.main import app
        uvicorn.run(
            app,
            host=args.host or cfg.api_host,
            port=args.port or cfg.api_port,
        )
        return 0

    try:
        columns = load_columns(args.file)
    except FileReadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    repo = SQLiteRepository(cfg.sqlite_path)

    if args.cmd == "control-chart":
        return _run_control_chart(args, columns, cfg, repo)
    if args.cmd == "capability":
        return _run_capability(args, columns, cfg, repo)
    if args.cmd == "msa":
        return _run_msa(args, columns, cfg, repo)
    return 1


def _run_control_chart(args, columns, cfg, repo) -> int:
    from spc_core import establish, phase1_checklist

    frame = ingest(
        columns,
        value_col=args.value_col,
        subgroup_col=args.subgroup_col,
        sample_size_col=args.sample_size_col,
        opportunity_col=args.opportunity_col,
    )
    cmap = frame.column_map
    if cmap.value_col is None:
        print("Error: could not detect measurement column", file=sys.stderr)
        return 1

    ct = ChartType(args.chart_type) if args.chart_type else None
    pipeline = establish(
        columns[cmap.value_col],
        subgroup_ids=columns.get(cmap.subgroup_col) if cmap.subgroup_col else None,
        sample_sizes=columns.get(cmap.sample_size_col) if cmap.sample_size_col else None,
        opportunities=columns.get(cmap.opportunity_col) if cmap.opportunity_col else None,
        chart_type=ct,
        ruleset=args.ruleset or cfg.ruleset,
        acf_threshold=cfg.acf_threshold,
    )
    checklist = phase1_checklist(pipeline, min_subgroups=cfg.min_phase1_points)
    result = pipeline.chart
    report = SPCReport.from_chart_result(
        result,
        normality=pipeline.normality,
        autocorrelation=pipeline.autocorrelation,
        source_file=args.file,
        gates=pipeline.gates,
        checklist=checklist,
    )
    report_dict = report.model_dump(mode="json")
    repo.save_limits(report_dict["limits"], report.limits.version, report.chart_type.value)
    run_id = repo.save_run("control_chart", report_dict, limits_version=report.limits.version,
                           source_file=args.file)

    if args.html:
        save_html(render_control_chart_html(report), args.html)
        print(f"HTML report: {args.html}")

    if args.json:
        print(json.dumps({"run_id": run_id, **report_dict}, indent=2, default=str))
    else:
        print(f"Chart: {result.chart_type.value}")
        print(f"Limits version: {result.limits.version}")
        print(f"Points: {len(result.plotted_values)}")
        print(f"Signals: {result.out_of_control_count}")
        print(f"Run ID: {run_id}")
        print(f"Checklist passed: {checklist.get('passed')}")
        for s in result.signals[:10]:
            print(f"  [{s.rule_id}] point {s.index}: {s.description}")
    return 0


def _run_capability(args, columns, cfg, repo) -> int:
    frame = ingest(columns, value_col=args.value_col, subgroup_col=args.subgroup_col)
    cmap = frame.column_map
    if cmap.value_col is None:
        print("Error: could not detect measurement column", file=sys.stderr)
        return 1
    values = [float(v) for v in columns[cmap.value_col] if v is not None]
    subgroups = None
    if cmap.subgroup_col and cmap.subgroup_col in columns:
        from spc_core.limits import build_subgroups
        subgroups = build_subgroups(values, columns[cmap.subgroup_col])

    normality = check_normality(values)
    result = capability_analysis(
        values, usl=args.usl, lsl=args.lsl, target=args.target, subgroups=subgroups
    )
    report = CapabilityReport.from_capability(result, normality=normality, source_file=args.file)
    report_dict = report.model_dump(mode="json")
    run_id = repo.save_run("capability", report_dict, source_file=args.file)

    if args.html:
        save_html(render_capability_html(report), args.html)
        print(f"HTML report: {args.html}")

    if args.json:
        print(json.dumps({"run_id": run_id, **report_dict}, indent=2, default=str))
    else:
        print(f"Method: {result.method}")
        print(f"Cpk: {result.cpk}  Ppk: {result.ppk}")
        print(f"DPMO: {result.observed_dpmo:.1f}  Sigma level: {result.sigma_level}")
        print(f"Rating: {result.rating}")
        print(f"Normality: is_normal={normality.is_normal} shapiro_p={normality.shapiro_p}")
        print(f"Run ID: {run_id}")
    return 0


def _run_msa(args, columns, cfg, repo) -> int:
    frame = ingest(
        columns,
        value_col=args.measurement_col,
        part_col=args.part_col,
        operator_col=args.operator_col,
        reference_col=args.reference_col,
    )
    cmap = frame.column_map
    st = args.study_type
    if st is None:
        if cmap.part_col and cmap.operator_col:
            st = "gage_rr"
        elif cmap.reference_col:
            refs = set(columns[cmap.reference_col])
            st = "linearity" if len(refs) > 1 else "bias"
        else:
            st = "stability"

    if st == "gage_rr":
        fn = gage_rr_anova if args.method == "anova" else gage_rr_range
        result = fn(columns[cmap.part_col], columns[cmap.operator_col],
                    columns[cmap.value_col], tolerance=args.tolerance)
        report = MSAReport.from_gage_rr(result, source_file=args.file)
        summary = f"GRR%={result.grr_percent:.1f} NDC={result.ndc} {result.acceptability}"
    elif st == "bias":
        result = bias_study(columns[cmap.value_col], columns[cmap.reference_col])
        report = MSAReport.from_bias(result, source_file=args.file)
        summary = f"bias={result.mean_bias:.4f} p={result.p_value:.4f}"
    elif st == "linearity":
        result = linearity_study(columns[cmap.value_col], columns[cmap.reference_col])
        report = MSAReport.from_linearity(result, source_file=args.file)
        summary = f"slope={result.slope:.4f} R²={result.r_squared:.3f}"
    else:
        result = stability_study(columns[cmap.value_col])
        report = MSAReport.from_stability(result, source_file=args.file)
        summary = f"stable={result.is_stable} OOC={result.out_of_control_points}"

    report_dict = report.model_dump(mode="json")
    run_id = repo.save_run("msa", report_dict, source_file=args.file)

    if args.html:
        save_html(render_msa_html(report), args.html)
        print(f"HTML report: {args.html}")

    if args.json:
        print(json.dumps({"run_id": run_id, **report_dict}, indent=2, default=str))
    else:
        print(f"Study: {report.study_type}")
        print(summary)
        # Always expose grr_percent for gage_rr (fixes the legacy key mismatch)
        if st == "gage_rr":
            print(f"grr_percent: {result.grr_percent}")
        print(f"Run ID: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
