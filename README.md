# Agentic tool for statistical process control

A Quality Management System for **Statistical Process Control (SPC)**, **Measurement System Analysis (MSA)**, and **Process Capability** — powered by agents.

---

## Overview

Quality engineers spend too much time clicking through interfaces and manually interpreting charts.
**SPC Quality AI** automates the tedious parts of quality analysis just upload your data and ask questions in plain English.

No more:

* Guessing which control chart to use
* Manually calculating control limits
* Repeating the same MSA interpretations

Instead, the AI agents handle:

* Automatic detection of the correct chart type
* Identification of out-of-control points
* Plain-language interpretation of results
* Professional report generation

They’re not perfect, but they’re remarkably efficient assistants for modern quality engineers.

---

## Features

###  Statistical Process Control (SPC)

* **Control Charts:** I-MR, Xbar-R, Xbar-S, P, NP, C, and U charts
* **Automatic Chart Selection** based on dataset characteristics
* **Out-of-Control Detection** with statistical rules
* **Data Quality Validation** before analysis
* **Interactive Visualizations** via Plotly

###  Measurement System Analysis (MSA)

* **Gage R&R Studies** (ANOVA method)
* **Bias Studies** with significance testing
* **Linearity Studies** for accuracy verification
* **Stability Studies** for long-term consistency
* **Comprehensive MSA Reports** with interpretation

###  Process Capability Analysis

* **Capability Indices:** Cp, Cpk, Pp, Ppk, Cpm
* **Normality Testing:** Anderson–Darling, Shapiro–Wilk, Kolmogorov–Smirnov
* **Process Centering and Variation Analysis**

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/M1ndSmith/ASPC.git
cd ASPC

# Create and activate virtual environment
python -m venv aspcvenv
source aspcvenv/bin/activate  # On Windows: spc_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp env.example .env
# Edit .env and add your GROQ_API_KEY (or key for your chosen LLM)
```

### 2. Configure LLM (Optional)

The system uses **Groq** by default (free tier available).
To switch providers, edit `agent_config/config.yaml`:

```yaml
llm:
  provider: "groq"        # or "openai", "anthropic"
  model: "llama-3.1-8b-instant"
```

### 3. Start the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Server available at: [http://localhost:8000](http://localhost:8000)

---

## Using the CLI

The CLI (`./spc`) provides the easiest way to interact with the agents.
It automatically activates the virtual environment.

### Basic Commands

```bash
# Check server health
./spc health

# Run control chart analysis
./spc chat control-charts "Analyze this data" -f your_data.csv

# Run MSA study
./spc chat msa "Run a Gage R&R study" -f measurement_data.csv

# Run capability analysis
./spc chat capability "Calculate Cp and Cpk with USL=10.5, LSL=9.5" -f process_data.csv
```

### Available Agents

| Agent            | Purpose                        |
| ---------------- | ------------------------------ |
| `control-charts` | Control chart and SPC analysis |
| `msa`            | Measurement System Analysis    |
| `capability`     | Process Capability Analysis    |

### CLI Options

| Option        | Short | Description                  | Default                 |
| ------------- | ----- | ---------------------------- | ----------------------- |
| `--file`      | `-f`  | CSV file to upload           | None                    |
| `--thread-id` | `-t`  | Conversation thread ID       | `cli_session`           |
| `--user-id`   | `-u`  | User identifier              | `cli_user`              |
| `--timeout`   |       | Request timeout (seconds)    | 120                     |
| `--verbose`   | `-v`  | Verbose output with metadata | False                   |
| `--output`    | `-o`  | Save response to JSON file   | None                    |
| `--url`       |       | API base URL                 | `http://localhost:8000` |

---

### CLI Examples

**Control Charts**

```bash
# Basic analysis
./spc chat control-charts "Analyze this data" -f data.csv

# Verbose output
./spc chat control-charts "Check for out-of-control points" -f data.csv -v

# Save results
./spc chat control-charts "Generate full report" -f data.csv -o report.json
```

**MSA Example**

```bash
./spc chat msa "Conduct a Gage R&R study" -f gage_data.csv
```

**Capability Example**

```bash
./spc chat capability "Assess capability with USL=10.5, LSL=9.5, target=10.0" -f process_data.csv
```

**Conversational Threads**

```bash
# Initial message
./spc chat control-charts "Analyze this data" -f data.csv -t analysis_001

# Follow-up queries using same thread
./spc chat control-charts "What are the main issues?" -t analysis_001
./spc chat control-charts "Give me recommendations" -t analysis_001
```

**Advanced Usage**

Custom API URL:

```bash
./spc --url http://remote-server:8000 chat msa "Run study" -f data.csv
```

Extended Timeout:

```bash
./spc chat control-charts "Analyze large dataset" -f big.csv --timeout 300
```

Batch Automation:

```bash
#!/bin/bash
for file in data/*.csv; do
  echo "Processing $file..."
  ./spc chat control-charts "Analyze and report" -f "$file" -t batch_$(date +%Y%m%d)
done
```

---

## API Usage

You can also call the API directly.

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/chat/control-charts",
    data={"message": "Analyze this data", "thread_id": "my_session", "user_id": "engineer1"},
    files={"file": open("data.csv", "rb")}
)
print(response.json()["response"])
```

### curl Example

```bash
curl -X POST http://localhost:8000/chat/control-charts \
  -F "message=Analyze this data" \
  -F "thread_id=session1" \
  -F "user_id=user1" \
  -F "file=@data.csv"
```

### Endpoints

| Method | Endpoint           | Description                                                |
| ------ | ------------------ | ---------------------------------------------------------- |
| `GET`  | `/`                | API overview                                               |
| `GET`  | `/health`          | Health check                                               |
| `POST` | `/chat/{agent_id}` | Chat with an agent (`control-charts`, `msa`, `capability`) |

---

## Data Format

Each CSV should include the relevant measurement columns.
The system will auto-detect column roles but supports explicit naming.

**Control Charts**

```csv
date,measurement
2024-01-01,10.2
2024-01-02,10.1
2024-01-03,9.9
```

**Subgroup Data**

```csv
subgroup,measurement
1,10.2
1,10.1
1,9.9
2,10.0
2,10.3
```

**MSA (Gage R&R)**

```csv
Part,Operator,Measurement,Trial
1,A,10.2,1
1,A,10.1,2
1,B,10.3,1
1,B,10.2,2
```

**Process Capability**

```csv
measurement
10.2
10.1
9.9
10.0
```

The system is tolerant of alternative column names and will infer structure automatically.

---

## Sample Data

### Control Charts

* `spc_individual_in_control.csv`
* `spc_individual_out_of_control.csv`
* `spc_subgroup_data.csv`
* `spc_np_chart_data.csv`
* `spc_c_chart_data.csv`

### MSA Studies

* `msa_gage_rr_excellent.csv`
* `msa_gage_rr_poor.csv`
* `msa_bias_study.csv`
* `msa_linearity_study.csv`
* `msa_stability_study.csv`

### Process Capability

* `capability_excellent.csv`
* `capability_off_center.csv`
* `capability_high_variation.csv`
* `capability_skewed_data.csv`

---

## Output

Each analysis produces:

* Detailed statistical summaries
* HTML report with charts
* Plain-language recommendations
* Out-of-control detection (SPC)
* Acceptance criteria (MSA)
* Capability indices (Cp, Cpk, Pp, Ppk)

Reports are stored in `temp_uploads/`.

---

## Project Structure

```
├── agent_config/               # LLM and agent settings
│   ├── config.yaml             # Model provider config
│   └── agent_prompts/          # Agent prompt templates
├── api/                        # FastAPI server
├── control_chart_system/       # SPC logic
├── msa_system/                 # MSA logic
├── process_capability_system/  # Capability analysis logic
├── data_samples/               # Example datasets
├── spc_cli.py                  # Command-line tool
```

---

## Example Workflow

```bash
# 1. Start the server
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# 2. Validate measurement system
./spc chat msa "Conduct Gage R&R study" -f gage_data.csv

# 3. If MSA passes, monitor process
./spc chat control-charts "Check process stability" -f process_data.csv

# 4. If process is stable, check capability
./spc chat capability "Calculate Cp and Cpk with USL=10.5, LSL=9.5" -f process_data.csv
```

---

## Use Cases

### Manufacturing

* Monitor process stability using control charts
* Validate measurement systems for precision
* Assess process capability for continuous improvement

### Research & Development

* Verify measurement reliability before experiments
* Ensure data integrity in prototypes or trials
* Automate statistical exploration of test results

---

## Conversation Memory

Agents retain context within a session (`--thread-id`), allowing natural multi-step discussions:

```bash
./spc chat control-charts "Analyze this" -f data.csv -t project_001
./spc chat control-charts "Explain those out-of-control points" -t project_001
./spc chat control-charts "Give improvement suggestions" -t project_001
```

---

## Reports example

<img width="877" height="771" alt="control_chart" src="https://github.com/user-attachments/assets/2f5409b3-1830-44e5-8d86-61fb2ef87c21" />

<img width="877" height="771" alt="msa_report" src="https://github.com/user-attachments/assets/898717fc-04ae-48d0-87a0-d5772a44a21c" />

<img width="874" height="819" alt="capability analysis" src="https://github.com/user-attachments/assets/b3d2545f-fc40-4ae5-80e8-1244b58b0f8e" />



---
## Contributing

This is a **work in progress** project.
Contributions are welcome code, documentation, or any ideas.

1. Open an issue describing your suggestion or bug
2. Fork and submit a pull request

---

## Credits

Built with:

* **Langchain** — Prebuilt agents
* **FastAPI** — REST API
* **Plotly** — Visualization
* **scipy**, **numpy**, **pandas** — Statistical backbone

---

**Disclaimer:**
This tool uses AI (LLMs) for statistical interpretation.
Always verify results with standard software and domain expertise before making decisions.
The AI assists quality engineers — it doesn’t replace them.


