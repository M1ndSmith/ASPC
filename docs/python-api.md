# Python library (`spc_core`)

Pure computation — no I/O, no FastAPI. Public surface: [`spc_core/__init__.py`](../spc_core/__init__.py).

Install:

```bash
uv pip install -e ".[dev]"
```

Demo data without CSV files:

```python
from sample_data import spc_individual, msa_gage_rr, capability
```

## Phase I (gated)

```python
from sample_data import spc_individual
from spc_core import establish, phase1_checklist

cols = spc_individual(in_control=True)
pipe = establish(cols["measurement"], ruleset="nelson", acf_threshold=0.2)

print(pipe.chart.chart_type, pipe.limits_version, pipe.chart_route, pipe.stopped)
for g in pipe.gates:
    print(f"[{g.status}] {g.step}: {g.reason}")

checklist = phase1_checklist(pipe, min_subgroups=25, phase2_enabled=False)
assert "items" in checklist

records = pipe.chart.to_records()  # list[SPCRecord]
```

Optional MSA gate inside `establish`:

```python
from sample_data import msa_gage_rr
from spc_core import establish

msa = msa_gage_rr(quality="excellent")
pipe = establish(
    spc_individual()["measurement"],
    msa_parts=msa["Part"],
    msa_operators=msa["Operator"],
    msa_measurements=msa["Measurement"],
    msa_tolerance=1.0,
)
```

## Charts directly

```python
from spc_core import analyze_control_chart, ChartType
from spc_core import ewma_chart, cusum_chart

result = analyze_control_chart(values, subgroup_ids=None, ruleset="nelson")
# or force type:
result = analyze_control_chart(values, chart_type=ChartType.XBAR_R, subgroup_ids=sids)

ew = ewma_chart(values, lam=0.2, L=3.0)
cu = cusum_chart(values, k=0.5, h=5.0)
```

Limit builders: `imr_limits`, `xbar_r_limits`, `xbar_s_limits`, `p_limits`, `np_limits`, `c_limits`, `u_limits`.

## Capability

```python
from sample_data import capability
from spc_core import capability_analysis, check_normality

values = capability(kind="excellent")["measurement"]
norm = check_normality(values)
cap = capability_analysis(values, usl=10.5, lsl=9.5, target=10.0)
print(cap.method, cap.cpk, cap.ppk, cap.sigma_level, cap.rating)
```

## MSA

```python
from sample_data import msa_gage_rr, msa_bias
from spc_core import gage_rr_anova, bias_study, ndc_gate, gage_resolution_gate

cols = msa_gage_rr(quality="excellent")
rr = gage_rr_anova(cols["Part"], cols["Operator"], cols["Measurement"])
print(rr.grr_percent, rr.ndc, rr.acceptability)
print(ndc_gate(rr.ndc))
print(gage_resolution_gate(0.01, tolerance=1.0))

b = bias_study(msa_bias()["Measurement"], msa_bias()["Reference"])
```

Continuous / streaming MSA:

```python
from spc_core import ContinuousMSA

msa = ContinuousMSA(reference=10.0, bias_threshold=0.2)
alert = msa.update(10.15)  # CalibrationAlert or None
```

## Phase II evaluation

Against **frozen** limits from Phase I:

```python
from spc_core import evaluate_batch, Phase2Evaluator
from adapters.stream import FileReplaySource, stream_evaluate
from sample_data import write_csv, spc_individual

pipe = establish(spc_individual(in_control=True)["measurement"])
limits = pipe.chart.limits

# Batch
signals = evaluate_batch(new_values, limits, ruleset="nelson")

# Stateful evaluator
ev = Phase2Evaluator(limits, ruleset="nelson")
for x in new_values:
    sigs = ev.update(x)

# File / stream adapter
path = write_csv(spc_individual(in_control=False), "examples/data/phase2.csv")
signals = stream_evaluate(FileReplaySource(path, "measurement"), limits)
```

## Reports

```python
from spc_core.report import SPCReport, CapabilityReport, MSAReport

report = SPCReport.from_chart_result(
    pipe.chart,
    normality=pipe.normality,
    autocorrelation=pipe.autocorrelation,
    gates=pipe.gates,
    checklist=checklist,
    include_records=True,
)
payload = report.model_dump(mode="json")
```

## Ingest helpers

```python
from spc_core import ingest, detect_columns
from adapters.io_files import load_columns

cols = load_columns("examples/data/spc_subgroup_data.csv")
frame = ingest(cols)  # frame.column_map.value_col, .subgroup_col, …
```

## Package extras

```bash
uv pip install -e ".[stream]"   # aiokafka, aiomqtt
uv pip install -e ".[tsdb]"     # sqlalchemy, alembic, asyncpg, psycopg
uv pip install -e ".[apps]"     # FastAPI stack
uv pip install -e ".[all]"
uv pip install -e ".[dev]"      # everything needed for tests + local apps
```

Next: [pipeline.md](pipeline.md) · [cli.md](cli.md) · [api.md](api.md)
