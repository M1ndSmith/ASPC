# SPC Quality AI

AI-powered quality management system for Statistical Process Control, MSA, and Process Capability analysis.

## What This Does

This is a set of AI agents that help with quality engineering tasks. Instead of manually running statistical analyses, you can just upload your data and ask questions in plain English. The agents handle the math, generate charts, and explain what's going on with your process.

Three main agents:
- **Control Charts**: Monitors process stability, detects out-of-control conditions
- **MSA**: Validates your measurement system (Gage R&R, bias, linearity, stability)
- **Capability**: Calculates Cp, Cpk, Pp, Ppk and tells you if your process can meet specs

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/M1ndSmith/AI-powered-SPC-quality-management-system.git
cd AI-powered-SPC-quality-management-system

# Create virtual environment
python -m venv spc_env
source spc_env/bin/activate  # On Windows: spc_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Configure LLM (Optional)

The system uses Groq by default (free tier available). To change:

Edit `agent_config/config.yaml`:
```yaml
llm:
  provider: "groq"  # or "openai", "anthropic"
  model: "llama-3.1-8b-instant"
```

### Start the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Server runs at `http://localhost:8000`

## Using the CLI

The easiest way to interact with the agents. The `./spc` wrapper script automatically activates the virtual environment.

### Basic Commands

```bash
# Check if the API server is running
./spc health

# Run a control chart analysis
./spc chat control-charts "Analyze this data" -f your_data.csv

# MSA study
./spc chat msa "Run a Gage R&R study" -f measurement_data.csv

# Capability analysis
./spc chat capability "Calculate Cp and Cpk with USL=10.5, LSL=9.5" -f process_data.csv
```

### Available Agents

- `control-charts` - Control chart analysis and SPC
- `msa` - Measurement System Analysis
- `capability` - Process Capability Analysis

### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--file` | `-f` | CSV file to upload | None |
| `--thread-id` | `-t` | Thread ID for conversation continuity | `cli_session` |
| `--user-id` | `-u` | User ID for tracking | `cli_user` |
| `--timeout` | | Request timeout in seconds | 120 |
| `--verbose` | `-v` | Show verbose output with metadata | False |
| `--output` | `-o` | Save response to JSON file | None |
| `--url` | | API base URL | `http://localhost:8000` |

### CLI Examples

**Control Charts Analysis:**
```bash
# Basic analysis
./spc chat control-charts "Analyze this data" -f data.csv

# With verbose output
./spc chat control-charts "Check for out-of-control points" -f data.csv -v

# Save response to file
./spc chat control-charts "Generate full report" -f data.csv -o report.json
```

**MSA - Gage R&R:**
```bash
./spc chat msa "Conduct a Gage R&R study" -f gage_data.csv
```

**Capability Analysis:**
```bash
./spc chat capability "Assess capability with USL=10.5, LSL=9.5, target=10.0" -f process_data.csv
```

**Conversational Follow-ups:**

Use the same thread ID to maintain conversation context:

```bash
# First message
./spc chat control-charts "Analyze this data" -f data.csv -t analysis_001

# Follow-up questions (same thread)
./spc chat control-charts "What are the main issues?" -t analysis_001
./spc chat control-charts "Give me recommendations" -t analysis_001
```

### Advanced CLI Usage

**Custom API URL:**
```bash
./spc --url http://remote-server:8000 chat msa "Run study" -f data.csv
```

**Extended Timeout for Large Datasets:**
```bash
./spc chat control-charts "Analyze complex data" -f large_file.csv --timeout 300
```

**Batch Processing:**
```bash
#!/bin/bash
for file in data/*.csv; do
    echo "Processing $file..."
    ./spc chat control-charts "Analyze and report" -f "$file" -t batch_$(date +%Y%m%d)
done
```

## API Usage

If you prefer to use the API directly:

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/chat/control-charts",
    data={
        "message": "Analyze this data for control charts",
        "thread_id": "my_session",
        "user_id": "engineer1"
    },
    files={"file": open("data.csv", "rb")}
)

print(response.json()["response"])
```

### curl

```bash
curl -X POST http://localhost:8000/chat/control-charts \
  -F "message=Analyze this data" \
  -F "thread_id=session1" \
  -F "user_id=user1" \
  -F "file=@data.csv"
```

### API Endpoints

- `GET /` - API info and available endpoints
- `GET /health` - Health check
- `POST /chat/{agent_id}` - Chat with an agent (agent_id: control-charts, msa, capability)

## Data Format

Your CSV should have columns for measurements. The agents will auto-detect most things, but you can be explicit:

**Control Charts:**
```csv
date,measurement
2024-01-01,10.2
2024-01-02,10.1
2024-01-03,9.9
```

For subgroup data:
```csv
subgroup,measurement
1,10.2
1,10.1
1,9.9
2,10.0
2,10.3
```

**MSA (Gage R&R):**
```csv
Part,Operator,Measurement,Trial
1,A,10.2,1
1,A,10.1,2
1,B,10.3,1
1,B,10.2,2
```

**Capability:**
```csv
measurement
10.2
10.1
9.9
10.0
```

The agents are smart about column detection. If your columns are named differently, they'll figure it out.

## What You Get

Each analysis generates:
- Detailed statistical results
- HTML report with charts
- Recommendations in plain English
- Out-of-control point detection (for control charts)
- Acceptance criteria (for MSA)
- Process capability indices (Cp, Cpk, Pp, Ppk)

Reports are saved to `temp_uploads/` directory.

## Project Structure

```
├── agent_config/          # LLM and agent configuration
│   ├── config.yaml        # LLM settings
│   └── agent_prompts/     # Agent prompts
├── api/                   # FastAPI server
├── control_chart_system/  # Control charts agent
├── msa_system/            # MSA agent
├── process_capability_system/  # Capability agent
├── data_samples/          # Example datasets
├── spc_cli.py            # Command-line tool
├── spc                    # CLI wrapper script
└── tests/                 # Test suite
```

## Example Workflow

```bash
# 1. Start the server
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# 2. Validate your measurement system
./spc chat msa "Conduct Gage R&R study" -f gage_data.csv

# 3. If MSA is good, monitor process with control charts
./spc chat control-charts "Check process stability" -f process_data.csv

# 4. If process is stable, assess capability
./spc chat capability "Calculate capability indices with USL=10.5, LSL=9.5" -f process_data.csv
```

## Configuration

### LLM Providers

Tested with:
- **Groq** (llama-3.1-8b-instant) - Free tier available, fast responses
- **OpenAI** (gpt-4, gpt-3.5-turbo)
- **Anthropic** (claude-3)

Edit `agent_config/config.yaml` to switch providers. Add your API key to `.env` file.

### Conversation Memory

Agents remember context within a thread. This lets you have natural conversations:

```bash
./spc chat control-charts "Analyze this" -f data.csv -t project_001
# Agent analyzes and responds

./spc chat control-charts "What caused those out-of-control points?" -t project_001
# Agent remembers previous analysis

./spc chat control-charts "Should I investigate this further?" -t project_001
# Agent provides recommendations based on full conversation
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html

# Test CLI functionality
python test_cli.py
```

## Performance

Typical response times:
- Simple queries: 1-5 seconds
- Control chart analysis: 5-30 seconds  
- Complex MSA/Capability: 30-60 seconds

Response time depends on:
- LLM provider speed (Groq is fastest)
- Data size (keep under 1000 points for best performance)
- Complexity of analysis

## Known Issues

- **First call is slower**: Model loading takes extra time (~5-10s)
- **Large datasets**: Files with >1000 points may timeout - subsample your data
- **Rate limits**: Groq free tier has limits - add 3-5 second delays between requests if needed
- **Thread initialization**: Very first test after server start may need a retry

These are minor and don't affect normal usage.

## Requirements

- Python 3.10+
- Internet connection (for LLM API calls)
- API key for your chosen LLM provider
- 2GB RAM minimum

## Troubleshooting

**"API connection error"**
- Check if server is running: `./spc health`
- Restart server: `uvicorn api.main:app --reload`

**"Request timed out"**
- Use `--timeout 180` for complex analyses
- Reduce data size if possible

**"Rate limit exceeded"**
- Add delays between requests
- Upgrade your LLM provider plan
- Switch to a different provider

## Why I Built This

Quality engineers spend too much time clicking through software and manually interpreting charts. I got tired of it. 

This automates the tedious parts:
- No more guessing which control chart to use
- No more manual calculation of control limits
- No more writing the same MSA interpretations over and over

The agents aren't perfect, but they're pretty good at:
- Detecting which chart type you need
- Finding out-of-control points
- Explaining results in plain language
- Generating professional reports you can share

Think of them as a junior quality engineer that works 24/7, never complains, and costs you nothing but an API key.

## What's Next

Possible improvements (contributions welcome):
- Add CUSUM and EWMA charts
- Support for multivariate SPC
- Integration with data historians (OSIsoft PI, etc.)
- Real-time monitoring mode
- Email alerts for out-of-control conditions
- Better visualization options

## Contributing

This is a working project, not perfectly polished. If you find bugs or have improvements:

1. Open an issue describing the problem
2. Fork and submit a PR
3. Make sure tests pass: `pytest tests/`

Code contributions, documentation improvements, and bug reports all welcome.

## License

MIT License - use it however you want.

## Credits

Built with:
- LangGraph for agent orchestration
- FastAPI for the API
- Plotly for charts
- scipy, numpy, pandas for statistics

## Contact

Issues and questions: Open a GitHub issue

---

**Disclaimer:** This uses AI/LLMs for analysis. Always verify critical decisions with proper statistical software and domain expertise. The agents are tools to assist, not replace, quality engineering judgment. Don't bet your job on what an LLM tells you without checking it first.
