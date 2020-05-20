#!/bin/bash
#SBATCH -J create_fred_tfcg
#SBATCH -p general
#SBATCH -o create_fred_tfcg_%j.txt
#SBATCH -e create_fred_tfcg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/create_graphs
time python create_fred.py -ft tfcg