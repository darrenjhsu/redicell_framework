#!/bin/bash

nvidia-cuda-mps-control -d

NUMA=`nvidia-smi topo -m | grep "GPU" | tail -1 | awk '{print $NF}'`
echo NUMA is $NUMA

numactl --cpunodebind=$NUMA python benchmark_full_ecoli.py --steps 30000000 --name cell1 &
numactl --cpunodebind=$NUMA python benchmark_full_ecoli.py --steps 30000000 --name cell2 &

wait
