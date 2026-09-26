#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# Prefer a known Python over whichever alias happens to be in the shell.
if [ ! -x .venv/bin/python ]; then
    if command -v python3.11 >/dev/null 2>&1; then
        python3.11 -m venv .venv
    elif [ -x /opt/homebrew/opt/python@3.11/bin/python3.11 ]; then
        /opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m venv .venv
    else
        echo "Install Python 3.11 or newer, then run this again."
        exit 1
    fi
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py --download-models
.venv/bin/python app.py --doctor
echo "Ready. Double-click Start EmotionDetection.command, or run .venv/bin/python app.py"
