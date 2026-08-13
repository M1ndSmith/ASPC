"""Plotly HTML renderer — consumes spc_core report models, produces HTML strings/files.

Rendering lives outside the core so the statistics library stays free of Plotly.
"""
from __future__ import annotations

from pathlib import Path

from spc_core.report import CapabilityReport, MSAReport, SPCReport


def render_control_chart_html(report: SPCReport, title: str | None = None) -> str:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for HTML reports. Install with: pip install aspc[render]"
        ) from exc

    title = title or f"Control Chart — {report.chart_type.value}"
    values = report.plotted_values
    seq = list(range(1, len(values) + 1))
    primary = report.limits.primary

    has_secondary = report.secondary_values is not None
    if has_secondary:
        fig = make_subplots(rows=2, cols=1, subplot_titles=[title, report.secondary_name or "Secondary"],
                            vertical_spacing=0.12)
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=[title])

    fig.add_trace(go.Scatter(x=seq, y=values, mode="lines+markers", name="Values",
                             line=dict(color="blue")), row=1, col=1)

    ucl = primary.ucl if not isinstance(primary.ucl, list) else primary.ucl
    lcl = primary.lcl if not isinstance(primary.lcl, list) else primary.lcl
    if isinstance(ucl, list):
        fig.add_trace(go.Scatter(x=seq, y=ucl, mode="lines", name="UCL",
                                 line=dict(color="red", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=seq, y=lcl, mode="lines", name="LCL",
                                 line=dict(color="red", dash="dash")), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=seq, y=[ucl] * len(seq), mode="lines", name="UCL",
                                 line=dict(color="red", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=seq, y=[lcl] * len(seq), mode="lines", name="LCL",
                                 line=dict(color="red", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=seq, y=[primary.center] * len(seq), mode="lines", name="CL",
                             line=dict(color="green")), row=1, col=1)

    if report.signals:
        ooc_idx = sorted({s.index for s in report.signals})
        ooc_x = [i + 1 for i in ooc_idx if i < len(values)]
        ooc_y = [values[i] for i in ooc_idx if i < len(values)]
        fig.add_trace(go.Scatter(x=ooc_x, y=ooc_y, mode="markers", name="OOC",
                                 marker=dict(color="red", size=10, symbol="x")), row=1, col=1)

    if has_secondary and report.secondary_values is not None:
        sec = report.secondary_values
        sec_seq = list(range(1, len(sec) + 1))
        fig.add_trace(go.Scatter(x=sec_seq, y=sec, mode="lines+markers", name=report.secondary_name,
                                 line=dict(color="green")), row=2, col=1)
        # secondary component if present
        comps = report.limits.components
        sec_key = report.secondary_name
        if sec_key and sec_key in comps:
            scomp = comps[sec_key]
            fig.add_trace(go.Scatter(x=sec_seq, y=[scomp.ucl] * len(sec_seq) if not isinstance(scomp.ucl, list) else scomp.ucl,
                                     mode="lines", name="Sec UCL",
                                     line=dict(color="red", dash="dash")), row=2, col=1)

    fig.update_layout(height=600 if has_secondary else 400, showlegend=True)

    signal_html = ""
    if report.signals:
        items = "".join(
            f"<li>[{s.rule_id}] point {s.index}: {s.description} (value={s.value:.4f})</li>"
            for s in report.signals
        )
        signal_html = f"<h2>Signals ({len(report.signals)})</h2><ul>{items}</ul>"
    else:
        signal_html = "<h2>Signals</h2><p>No out-of-control signals detected.</p>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:40px;}} table{{border-collapse:collapse;}}
td,th{{border:1px solid #ddd;padding:8px;}}</style></head><body>
<h1>{title}</h1>
<p>Phase: {report.phase.value} | Limits version: {report.limits.version} |
Subgroup size: {report.subgroup_size} | Points: {len(values)}</p>
{pio.to_html(fig, include_plotlyjs="cdn", full_html=False)}
{signal_html}
</body></html>"""


def render_capability_html(report: CapabilityReport, title: str = "Process Capability Report") -> str:
    r = report.result
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in r.items()
        if k != "notes" and v is not None
    )
    norm = ""
    if report.normality:
        n = report.normality
        norm = (
            f"<h2>Normality</h2><p>is_normal={n.get('is_normal')} | "
            f"shapiro_p={n.get('shapiro_p')} | anderson_stat={n.get('anderson_stat')} | "
            f"{n.get('recommendation')}</p>"
        )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:40px;}}
table{{border-collapse:collapse;}} td,th{{border:1px solid #ddd;padding:8px;}}</style>
</head><body>
<h1>{title}</h1>
<p>Method: {r.get('method')} | Rating: {r.get('rating')}</p>
{norm}
<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>
</body></html>"""


def render_msa_html(report: MSAReport, title: str | None = None) -> str:
    title = title or f"MSA Report — {report.study_type}"
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in report.result.items()
        if k != "detail"
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:40px;}}
table{{border-collapse:collapse;}} td,th{{border:1px solid #ddd;padding:8px;}}</style>
</head><body>
<h1>{title}</h1>
<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>
</body></html>"""


def save_html(html: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p
