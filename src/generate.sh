#!/bin/bash
source /cornn_tracotgraphy/venv/bin/activate
which python
python --version
python /cornn_tractography/src/generate.py "$@"
deactivate