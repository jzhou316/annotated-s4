#!/bin/bash

set -e
set -o pipefail


DEBUG=0


DATASET=cifar-jpeg-dct-stack-classification

MODEL=s4
D_MODEL=512
N_LAYERS=6
LR=0.005    # DO NOT use 5e-3, otherwise the checkpoints_dir would be different from the Python training script
BSZ=64
DP=0.25
EP=500
SUFFIX=""

EP=500
SUFFIX="1stlaylmax64"
