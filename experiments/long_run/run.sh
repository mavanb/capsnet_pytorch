#!/usr/bin/env bash

cd ../..

seed=$1

python train_capsnet.py --model_name mnist_:700epoch: --general_conf experiments/long_run/gen.conf --capsule_conf experiments/long_run/caps.conf --seed $seed

