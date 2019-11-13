#!/bin/bash

#SBATCH -J create_co-occur_tfcg
#SBATCH -p general
#SBATCH -o create_co-occur_tfcg_%j.txt
#SBATCH -e create_co-occur_tfcg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_co-occur.py -ft tfcg_co
time python create_co-occur.py -ft ffcg_co
time python create_co-occur.py -ft ufcg_co