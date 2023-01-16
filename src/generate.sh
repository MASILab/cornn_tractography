#!/bin/bash

export CORNN_DIR=/home-local/cornn_tractography
export SCIL_DIR=~/Apps/scilpy

source $CORNN_DIR/venv/bin/activate
python $CORNN_DIR/src/generate.py $@
deactivate