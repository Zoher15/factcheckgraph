#!/bin/bash

#SBATCH -J create_co_occur
#SBATCH -p general
#SBATCH -o create_co_occur_%j.txt
#SBATCH -e create_co_occur_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
errcho(){ >&2 echo $@; }
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/create_graphs/
errcho create graphs
time python create_co_occur.py -ft tfcg_co
time python create_co_occur.py -ft ffcg_co
time python create_co_occur.py -ft ufcg_co
errcho calculate stats
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/process_graphs/
time python calculate_stats.py -gc co_occur -gt tfcg_co
time python calculate_stats.py -gc co_occur -gt ffcg_co
time python calculate_stats.py -gc co_occur -gt ufcg_co
time python compile_stats.py