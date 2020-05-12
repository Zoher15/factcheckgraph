#!/bin/bash
#SBATCH -J extract_claims
#SBATCH -p general
#SBATCH -o extract_claims.txt
#SBATCH -e extract_claims.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7:00:00
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/create_graphs
time python extract_claims.py