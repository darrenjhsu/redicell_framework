#!/bin/bash

nvidia-cuda-mps-control -d

CUDA=$1
NUMA=$2
CONC=$3

CUDA_VISIBLE_DEVICES=$CUDA numactl --cpunodebind=$NUMA python benchmark_full_ecoli.py --steps 30000000 --name ${CONC}uM_1 --conc ${CONC} --out_freq 5 > ${CONC}uM_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=$CUDA numactl --cpunodebind=$NUMA python benchmark_full_ecoli.py --steps 30000000 --name ${CONC}uM_2 --conc ${CONC} --out_freq 5 > ${CONC}uM_2.log 2>&1 &

wait
