#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  📝 IBMCloud MCP Server - Bash script file.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Author: Maxim Shelepov
# Description: Initializes and spins up local development server for the docs site.
# Is referenceable in Makefile.

VENV_DIR="../venv"

# Verify that python is installed.
which python3 &> /dev/null
IS_PYTHON3_INSTALLED=$?
which python &> /dev/null
IS_PYTHON_INSTALLED=$?

if [[ $IS_PYTHON3_INSTALLED -ne 0 && $IS_PYTHON_INSTALLED -ne 0 ]]; then
  echo "⛔️ Python is a prerequisite. Please install now."
  exit 1
fi

# Check if virtual environment directory exists.
VENV_DIR_NOT_EXISTS=$( [ ! -d "$VENV_DIR" ] && echo true || echo false)

# Verify that a virtual environment has been created, if not create one.
if $VENV_DIR_NOT_EXISTS; then
  echo "💡 Virtual environment being created."

  which python3 &> /dev/null
  if [ $? -eq 0 ]; then
    python3 -m venv ../venv
  else
    python -m venv ../venv
  fi
fi

# Activate the virtual environment.
echo "💡 Starting local python environment.";
source "$VENV_DIR"'/bin/activate'

# If virtual environment was freshly created, install dependencies.
if $VENV_DIR_NOT_EXISTS; then
  echo "💡 Installing pip dependencies."
  pip install -r ../.github/workflows/mkdocs-requirements.txt
fi

echo "✅ Starting docs on localhost."
mkdocs serve --watch-theme -w mkdocs.yml
