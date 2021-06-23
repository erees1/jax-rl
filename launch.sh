#!/usr/bin/env bash
set -euo pipefail

# Add cuda to path
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

P_DIR=$(dirname $0)
VENV=$P_DIR/venv/bin/activate

# Options
env='CartPole-v1'
agent='ddqn'
train_eps=100
lr=0.01
df=0.99
render=False
test_eps=30
n_layers=2
batch_size=256
epsilon_hlife=1500

# Sweeps
seeds=(1 2 3)

# This will parse command line arguments and overwrite thoose above 
. scripts/parse_options.sh

# Set up output directory
save_dir=out/$env/$agent/EPS$train_eps-LR$lr-DF$df
mkdir -p "$save_dir"

for seed in ${seeds[@]}; do

cat <<EOF >"${save_dir}"/launch.qsh
#!/bin/bash

source $VENV

python3 src/run.py \
    --agent $agent \
    --train_eps $train_eps \
    --n_layers $n_layers \
    --seed $seed \
    --test_eps $test_eps \
    --lr $lr \
    --batch_size $batch_size \
    --epsilon_hlife $epsilon_hlife \
    --save_dir $save_dir/$seed \
    --discount_factor $df \
    --render $render
EOF

# Execute launch.qsh
chmod +x "${save_dir}"/launch.qsh
"${save_dir}/launch.qsh"
done

true;
