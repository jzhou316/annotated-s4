#!/bin/bash

set -e
set -o pipefail

##### USAGE:
# bash $0 $gpu_id (e.g. bash run_cifar10.sh 3)

gpu=${1:-3}

dataset=cifar-jpeg-huffdec-wzeros-stream-classification
model=s4
d_model=512
n_layers=6
# lr=0.005    # DO NOT use 5e-3, otherwise the checkpoints_dir would be different from the Python training script
lr=0.01
bsz=32
dp=0.25
ep=1000
suffix=""
if [[ ! -z $suffix ]]; then
    suf="-{$suffix}"
else
    suf=""
fi

checkpoints_dir=checkpoints/${dataset}/${model}-lay${n_layers}-d${d_model}-lr${lr}-bsz${bsz}-dp${dp}-ep${ep}${suf}
# this is consistent with inside the python training script
log_file=${checkpoints_dir}/train.log


# check if model checkpoints folder already exists
if [[ -d ${checkpoints_dir} ]]; then
    echo -n "The model checkpoints folder already exists [${checkpoints_dir}] - do you want to overwrite (y/n)? "
    read answer
    if [[ "$answer" == [Nn]* ]]; then
        echo "exiting"
        exit 0
    fi
    echo -e "Removing existing checkpoints (otherwise there will be flax error)\n"
    rm -r $checkpoints_dir
fi

# create directory for logging
mkdir -p ${checkpoints_dir}

echo "log saved at $log_file"


CUDA_VISIBLE_DEVICES=$gpu python -m s4.train \
    --dataset $dataset \
    --model $model \
    --d_model $d_model \
    --n_layers $n_layers \
    --bsz $bsz \
    --epoch $ep \
    --lr $lr \
    --p_dropout $dp \
    --use_wandb True \
    --suffix "$suffix" \
    &> $log_file &
