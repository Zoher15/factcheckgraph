#!/bin/bash
#SBATCH -J compile_fred_neighbors
#SBATCH -p general
#SBATCH -o compile_fred_neighbors_%j.txt
#SBATCH -e compile_fred_neighbors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=05:00:00
errcho(){ >&2 echo $@; }
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
################################################################
errcho fetching
time python fetch_fred.py -ft tfcg -cpu 48 -p
time python fetch_fred.py -ft ffcg -cpu 48 -p 
################################################################
errcho compiling
time python compile_fred.py -ft ufcg -cpu 48 -n 100
time python compile_fred.py -ft ufcg -cpu 48 -n 200
time python compile_fred.py -ft ufcg -cpu 48 -n 300
time python compile_fred.py -ft ufcg -cpu 48 -n 400
################################################################
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
################################################################
errcho find_shortest_paths
time python find_shortest_paths.py -st ufcg -ft tfcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -n 100
time python find_shortest_paths.py -st ufcg -ft ffcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -n 100
time python order_paths.py -fcg fred -ft ufcg
################################################################
errcho plot graphs 
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/plot_graphs/
time python plot_sp.py -fcg fred -ft ufcg -pt roc
time python plot_sp.py -fcg fred -ft ufcg -pt dist
# time python compile_fred.py -ft ufcg -cpu 48 -n 50
# time python compile_fred.py -ft ufcg -cpu 48 -n 60
# time python compile_fred.py -ft ufcg -cpu 48 -n 70
# time python compile_fred.py -ft ufcg -cpu 48 -n 80
# time python compile_fred.py -ft ufcg -cpu 48 -n 90
# time python compile_fred.py -ft ufcg -cpu 48 -n 100

