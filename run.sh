#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=72:00:00	
#$ -N abci-tutorial
#$ -cwd

source ~/.bashrc
conda activate abci
bash exp.sh
