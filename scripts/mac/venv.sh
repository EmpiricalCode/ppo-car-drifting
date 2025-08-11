#!/bin/bash

# Get the parent directory of the script location
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Creating virtual environment in $PARENT_DIR"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment
python3 -m venv "$PARENT_DIR/venv"

# Activate the virtual environment
source "$PARENT_DIR/venv/bin/activate"

# Check if requirements.txt exists
if [ -f "$PARENT_DIR/requirements.txt" ]; then
    echo "Installing packages from requirements.txt"
    pip install -r "$PARENT_DIR/requirements.txt"
else
    echo "Warning: requirements.txt not found in $PARENT_DIR"
fi

echo "Virtual environment setup complete."
echo "To activate the virtual environment, run:"
echo "source $PARENT_DIR/venv/bin/activate"