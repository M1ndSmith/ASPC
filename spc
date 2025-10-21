#!/bin/bash
# SPC CLI Wrapper Script
# Automatically activates virtual environment and runs the CLI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -f "spc_env/bin/activate" ]; then
    source spc_env/bin/activate
fi

# Run the CLI with all arguments
python spc_cli.py "$@"

