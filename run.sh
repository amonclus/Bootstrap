#!/usr/bin/env bash
set -euo pipefail

# Move to the repository root, regardless of where the script is launched from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON_BIN=""

# Find a Python 3 interpreter
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Error: Python 3 is not installed or not available in PATH."
    exit 1
fi

# Create the virtual environment if it does not exist
if [[ ! -d "$VENV_DIR" ]]; then
    echo "No virtual environment found. Creating $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Using existing virtual environment at $VENV_DIR"
fi

# Activate the virtual environment
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Ensure pip is available and up to date inside the venv
python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install --upgrade pip

# Install dependencies into the virtual environment
python -m pip install -r src/requirements.txt

# Launch the Streamlit app
exec python -m streamlit run src/app.py
