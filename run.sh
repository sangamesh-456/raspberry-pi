#!/usr/bin/env bash
# Runs a script with the folder's own Python environment:   ./run.sh run_ball_tracking.py [args]
cd "$(dirname "$0")"
exec ./venv/bin/python "$@"
