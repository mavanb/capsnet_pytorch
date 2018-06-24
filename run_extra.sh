#!/usr/bin/env bash

# done 402

python train_capsnet.py --model_name arch_4,8\;10,12\;10,16 --arch 4,8\;10,12\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
python train_capsnet.py --model_name arch_4,8\;50,12\;10,16 --arch 4,8\;50,12\;10,16  --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf

python train_capsnet.py --model_name arch_2,8\;50,10\;15,12\;10,16 --arch 2,8\;50,10\;15,12\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
python train_capsnet.py --model_name arch_2,8\;50,10\;15,12\;10,16 --arch 2,8\;50,10\;15,12\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
