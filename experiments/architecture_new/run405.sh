#!/usr/bin/env bash
source activate py364


cd ../..

seed=$1


sparse_list=(
    nodes_topk_0.3-0.3
    none
)

arch_list=(
    8\,8\;60\,16\;10\,16
    8\,8\;30\,16\;10\,16
    8\,8\;10\,16\;10\,16
    16\,8\;60\,16\;10\,16
    16\,8\;30\,16\;10\,16  #too big for memory if none
    16\,8\;10\,16\;10\,16
    32\,8\;30\,16\;10\,16
    32\,8\;10\,16\;10\,16
)
#    8\,8\;120\,16\;10\,16 : to big for memory
#     : done for mnist
for dataset in cifar10
do
    for arch in "${arch_list[@]}"
    do
        for sparse in "${sparse_list[@]}"
        do
            python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:${sparse}:${arch}:${seed}: --general_conf experiments/architecture_new/gen.conf --capsule_conf experiments/architecture_new/caps.conf --seed $seed --architecture ${arch}
        done
    done
done

#prun -t $1:$2:00 -v -np 1 -native "-C gpunode" ./run.sh



