#!/bin/bash
source /cornn_tracotgraphy/venv/bin/activate
python /cornn_tractography/src/generate.py "$@"
deactivate