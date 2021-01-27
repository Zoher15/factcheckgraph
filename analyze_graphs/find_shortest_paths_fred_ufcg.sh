#!/bin/bash
#SBATCH -J find_shortest_paths_fred_ufcg
#SBATCH -p general
#SBATCH -o find_shortest_paths_fred_ufcg%j.txt
#SBATCH -e find_shortest_paths_fred_ufcg%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=58G
#SBATCH --time=1:00:00
errcho(){ >&2 echo $@; }
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
################################################################
errcho find_shortest_paths
time python find_shortest_paths.py -st ufcg -ft tfcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -n 100
time python find_shortest_paths.py -st ufcg -ft ffcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -n 100
time python order_paths.py -fcg fred -ft ufcg
# time python order_paths.py -fcg fred
# ################################################################
# # errcho find baseline
# # time python find_baseline.py -bt knn -cpu 48 -mp roberta-base-nli-stsb-mean-tokens
# # time python find_baseline.py -bt all -mp roberta-base-nli-stsb-mean-tokens
# ################################################################
# errcho plot graphs 
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/plot_graphs/
time python plot_sp.py -fcg fred -ft ufcg -pt roc
time python plot_sp.py -fcg fred -ft ufcg -pt dist
# time python plot_sp.py -fcg fred -pt roc
# time python plot_sp.py -fcg fred -pt dist
