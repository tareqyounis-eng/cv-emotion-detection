#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    if ! /bin/bash setup.sh; then
        read -r -p "Setup failed. Press Return to close. " _reply
        exit 1
    fi
fi
if ! .venv/bin/python -c 'import cv2, numpy' 2>/dev/null; then
    if ! /bin/bash setup.sh; then
        read -r -p "Setup failed. Press Return to close. " _reply
        exit 1
    fi
fi
if ! .venv/bin/python app.py --download-models; then
    read -r -p "Could not prepare the models. Press Return to close. " _reply
    exit 1
fi
set +e
.venv/bin/python app.py "$@"
app_status=$?
set -e
if [ "$app_status" -ne 0 ]; then
    if [ "$app_status" -eq 134 ]; then
        echo "The native window could not start. Run this launcher from Finder or a normal Terminal window."
        echo "See VALIDATION.md for the restricted-session display limitation."
    else
        echo "Check Camera permission for Terminal in System Settings > Privacy & Security."
    fi
    read -r -p "Press Return to close. " _reply
    exit "$app_status"
fi
