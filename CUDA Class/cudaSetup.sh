#!/usr/bin/env bash

module load cuda nsight-compute nsight-systems
alias lsfrun='bsub -W 10 -nnodes 1 -P csc434 -Is jsrun -n1 -a1 -c1 -g1 '
alias lsfrunnsight='bsub -W 10 -nnodes 1 -P csc434 -Is jsrun -n1 -a1 -c1 -g1 nv-nsight-cu-cli '
alias lsfrunnsys='bsub -W 10 -nnodes 1 -P csc434 -Is jsrun -n1 -a1 -c1 -g1 nsys profile --stats=true '
