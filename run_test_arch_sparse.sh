#!/usr/bin/env bash

#  done on 432

python train_capsnet.py --model_name arch_8,8\;10,16\;10,16_sparse --arch 8,8\;10,16\;10,16 --dataset cifar10 --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk
python train_capsnet.py --model_name arch_8,8\;50,16\;10,16_sparse --arch 8,8\;50,16\;10,16 --dataset cifar10 --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk
python train_capsnet.py --model_name arch_8,8\;120,16\;10,16_sparse --arch 8,8\;120,16\;10,16 --dataset cifar10 --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk

python train_capsnet.py --model_name arch_16,8\;10,16\;10,16_sparse --arch 16,8\;10,16\;10,16 --dataset cifar10 --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk
python train_capsnet.py --model_name arch_16,8\;30,16\;10,16_sparse --arch 16,8\;30,16\;10,16 --dataset cifar10 --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk
python train_capsnet.py --model_name arch_16,8\;50,16\;10,16_sparse --arch 16,8\;50,16\;10,16 --dataset cifar10 --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk

python train_capsnet.py --model_name arch_32,8\;30,16\;20,16\;10,16_sparse --arch 32,8\;30,16\;20,16\;10,8 --dataset cifar10 --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk
python train_capsnet.py --model_name arch_16,8\;30,16\;20,16\;10,16_sparse --arch 16,8\;30,16\;20,16\;10,16 --dataset cifar10  --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk
python train_capsnet.py --model_name arch_8,8\;30,16\;20,16\;10,16_sparse --arch 8,8\;30,16\;20,16\;10,16 --dataset cifar10  --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk
python train_capsnet.py --model_name arch_8,8\;100,16\;20,16\;10,16_sparse --arch 8,8\;100,16\;20,16\;10,16 --dataset cifar10  --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk

python train_capsnet.py --model_name arch_32,8\;10,16\;10,16_sparse --arch 32,8\;10,16\;10,16 --dataset cifar10 --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk
python train_capsnet.py --model_name arch_32,8\;20,16\;10,16_sparse --arch 32,8\;20,16\;10,16 --dataset cifar10  --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk
python train_capsnet.py --model_name arch_32,8\;30,16\;10,16_sparse --arch 32,8\;30,16\;10,16  --dataset cifar10 --general_conf configurations/gen_arch.conf --capsule_conf configurations/caps_arch.conf --sparsify nodes_topk



