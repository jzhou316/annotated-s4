#!/bin/bash

set -e
set -o pipefail

##### USAGE:
# bash $0 $config_file $gpu_id (e.g. bash run_cifar10.sh configs/config_xxx.sh 3)


##### config
if [ -z "$1" ]; then
    :        # in this case, must provide $1 as "" empty; otherwise put "set -o nounset" below
else
    config=$1
    . $config    # $config should include its path
fi
# NOTE: when the first configuration argument is not provided, this script runs with default args below

gpu=${2:-1}


set -o nounset
# NOTE this should be set after "set_environment.sh", particularly for the line
#      eval "$(conda shell.bash hook)"


gpu=${2:-3}

debug=${DEBUG:-1}    # debug mode for development: no logging, iteractive run in console

##### default configs
dataset=${DATASET:-cifar-classification}
model=${MODEL:-s4}
d_model=${D_MODEL:-512}
n_layers=${N_LAYERS:-6}
lr=${LR:-0.005}    # DO NOT use 5e-3, otherwise the checkpoints_dir would be different from the Python training script
bsz=${BSZ:-64}
dp=${DP:-0.25}
ep=${EP:-100}
suffix=${SUFFIX:-""}
if [[ ! -z $suffix ]]; then
    suf="-$suffix"
else
    suf=""
fi

checkpoints_dir=checkpoints/${dataset}/${model}-lay${n_layers}-d${d_model}-lr${lr}-bsz${bsz}-dp${dp}-ep${ep}${suf}
# this is consistent with inside the python training script
log_file=${checkpoints_dir}/train.log


# ===== run training

if [[ $debug == "0" ]]; then

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


else

# in debug mode: print out the information, no wandb logging, and exit epoch 3

CUDA_VISIBLE_DEVICES=$gpu python -m s4.train \
    --dataset $dataset \
    --model $model \
    --d_model $d_model \
    --n_layers $n_layers \
    --bsz $bsz \
    --epoch 3 \
    --lr $lr \
    --p_dropout $dp \
    --use_wandb False \
    --suffix "${suffix}-debug"

fi
