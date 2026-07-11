#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -J collect
source /personal/soft/DeePTB/.venv/bin/activate
python run.py 

