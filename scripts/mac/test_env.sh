#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the parent directory (project root)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Activate the virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Run the main.py script
python "$PROJECT_ROOT/src/test_env.py" "$@"

# Deactivate the virtual environment when done
deactivate