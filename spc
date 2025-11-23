#!/bin/bash
# SPC Quality AI - Command Line Interface Wrapper
# This script provides the ./spc command interface mentioned in the README

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
if [ -f "$SCRIPT_DIR/aspcvenv/bin/activate" ]; then
    source "$SCRIPT_DIR/aspcvenv/bin/activate"
else
    echo "Error: Virtual environment not found at $SCRIPT_DIR/aspcvenv"
    echo "Please ensure the aspcvenv virtual environment is properly set up."
    exit 1
fi

# Run the CLI with all passed arguments
python "$SCRIPT_DIR/spc_cli.py" "$@"
