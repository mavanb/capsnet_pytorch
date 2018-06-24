#!/usr/bin/env bash

# done on 403

python train_capsnet.py --model_name arch_32,8\;30,16\;20,16\;10,16 --arch 32,8\;30,16\;20,16\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
python train_capsnet.py --model_name arch_16,8\;30,16\;20,16\;10,16 --arch 16,8\;30,16\;20,16\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
python train_capsnet.py --model_name arch_8,8\;30,16\;20,16\;10,16 --arch 8,8\;30,16\;20,16\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
python train_capsnet.py --model_name arch_8,8\;100,16\;20,16\;10,16 --arch 8,8\;100,16\;20,16\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf

#python train_capsnet.py --model_name arch_32,8\;10,16\;10,16 --arch 32,8\;10,16\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
#python train_capsnet.py --model_name arch_32,8\;20,16\;10,16 --arch 32,8\;20,16\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
#python train_capsnet.py --model_name arch_32,8\;30,16\;10,16 --arch 32,8\;30,16\;10,16  --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
#
#python train_capsnet.py --model_name arch_16,8\;10,16\;10,16 --arch 16,8\;10,16\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
#python train_capsnet.py --model_name arch_16,8\;30,16\;10,16 --arch 16,8\;30,16\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
#python train_capsnet.py --model_name arch_16,8\;50,16\;10,16 --arch 16,8\;50,16\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
#
#python train_capsnet.py --model_name arch_8,8\;10,16\;10,16 --arch 8,8\;10,16\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
#python train_capsnet.py --model_name arch_8,8\;50,16\;10,16 --arch 8,8\;50,16\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf
#python train_capsnet.py --model_name arch_8,8\;120,16\;10,16 --arch 8,8\;120,16\;10,16 --dataset cifar10 --sparsify None --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf










