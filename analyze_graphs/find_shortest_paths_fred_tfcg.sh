#!/bin/bash
#SBATCH -J find_shortest_paths_fred_tfcg
#SBATCH -p general
#SBATCH -o find_shortest_paths_fred_tfcg%j.txt
#SBATCH -e find_shortest_paths_fred_tfcg%j.err
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
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
################################################################
errcho find_shortest_paths
time python find_shortest_paths.py -st tfcg -ft tfcg -mp roberta-base-nli-stsb-mean-tokens -cpu 48 -n 50
time python find_shortest_paths.py -st tfcg -ft ffcg -mp roberta-base-nli-stsb-mean-tokens -cpu 48 -n 50
time python order_paths.py -fcg fred -ft tfcg
################################################################
errcho plot graphs 
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/plot_graphs/
time python plot_sp.py -fcg fred -ft tfcg -pt roc -pr n50
time python plot_sp.py -fcg fred -ft tfcg -pt dist -pr n50
# time python plot_sp.py -fcg fred -pt roc
# time python plot_sp.py -fcg fred -pt dist