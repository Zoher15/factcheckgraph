#!/bin/bash
#SBATCH -J find_shortest_paths_fred_undirected
#SBATCH -p general
#SBATCH -o find_shortest_paths_fred_undirected_%j.txt
#SBATCH -e find_shortest_paths_fred_undirected_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=58G
#SBATCH --time=16:00:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
time python embed.py -ft ffcg -fc fred -gt undirected
time python embed.py -ft tfcg -fc fred -gt undirected
time python embed.py -ft ffcg -fc fred -mp roberta-base-nli-stsb-mean-tokens -gt undirected
time python embed.py -ft tfcg -fc fred -mp roberta-base-nli-stsb-mean-tokens -gt undirected
time python find_shortest_paths.py -ft ffcg -fc fred -cpu 48 -gt undirected
time python find_shortest_paths.py -ft tfcg -fc fred -cpu 48 -gt undirected
time python find_shortest_paths.py -ft ffcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -gt undirected
time python find_shortest_paths.py -ft tfcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -gt undirected
