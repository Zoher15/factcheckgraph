#!/bin/bash
#SBATCH -J find_shortest_paths2_fred_directed
#SBATCH -p general
#SBATCH -o find_shortest_paths2_fred_directed_%j.txt
#SBATCH -e find_shortest_paths2_fred_directed_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=20G
#SBATCH --time=6:00:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
time python find_shortest_paths.py -ft ffcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -gt directed
time python find_shortest_paths.py -ft tfcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -gt directed