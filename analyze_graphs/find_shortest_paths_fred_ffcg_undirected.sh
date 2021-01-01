#!/bin/bash
#SBATCH -J find_shortest_paths_fred_ffcg_undirected
#SBATCH -p general
#SBATCH -o find_shortest_paths_fred_ffcg_undirected_%j.txt
#SBATCH -e find_shortest_paths_fred_ffcg_undirected_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=58G
#SBATCH --time=2:00:00
errcho(){ >&2 echo $@; }
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
errcho embed
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
time python embed.py -ft ffcg -fc fred -mp roberta-base-nli-stsb-mean-tokens -gt undirected
time python embed.py -ft tfcg -fc fred -mp roberta-base-nli-stsb-mean-tokens -gt undirected
################################################################
errcho find_shortest_paths
time python find_shortest_paths.py -st ffcg -ft ffcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -gt undirected
time python find_shortest_paths.py -st ffcg -ft tfcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -gt undirected
time python order_paths.py -fcg fred -ft ffcg
################################################################
errcho find baseline
time python find_baseline.py -bt knn -cpu 48 -mp roberta-base-nli-stsb-mean-tokens
time python find_baseline.py -bt all -mp roberta-base-nli-stsb-mean-tokens
################################################################
errcho plot graphs 
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/plot_graphs/
time python plot_sp.py -fcg fred -ft ffcg -pt roc
time python plot_sp.py -fcg fred -ft ffcg -pt dist
time python plot_sp.py -fcg fred -pt roc
time python plot_sp.py -fcg fred -pt dist
