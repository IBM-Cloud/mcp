#!/bin/bash
VENV_DIR="../venv"

if [ ! -d "$VENV_DIR" ]; then
  which python3
  if [ $? -eq 0 ]; then
    python3 -m venv ../venv
  else
    python -m venv ../venv
  fi
fi

source "$VENV_DIR"'/bin/activate'
pip install -r ../.github/workflows/mkdocs-requirements.txt
mkdocs serve --watch-theme -w mkdocs.yml
