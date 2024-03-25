#!/bin/bash
set -e
set -x


base_ckpt=$WHY_ATTACK_CKPT
base_data=$WHY_ATTACK_DATA

if [ -z "$base_ckpt" ]; then
    base_ckpt='.'
fi

if [ -z "$base_data" ]; then
    base_data='.'
fi

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export config=$1 # llama2 or vicuna
export data_offset=$2
export n_steps=$3

# Create results folder if it doesn't exist
if [ ! -d "${base_data}/results_n_steps_${n_steps}_${config}" ]; then
    mkdir "${base_data}/results_n_steps_${n_steps}_${config}"
    echo "Folder '${base_data}/results_n_steps_${n_steps}_${config}' created."
else
    echo "Folder '${base_data}/results_n_steps_${n_steps}_${config}' already exists."
fi

python -u ../main.py \
    --config="../configs/${config}.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_behaviors_train_split.csv" \
    --config.result_prefix="${base_data}/results_n_steps_${n_steps}_${config}/gcg_offset${data_offset}" \
    --config.n_train_data=10 \
    --config.data_offset=$data_offset \
    --config.n_steps=$n_steps \
    --config.test_steps=1 \
    --config.batch_size=256