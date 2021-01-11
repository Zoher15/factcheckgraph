#!/bin/bash
#SBATCH -J pipeline_shortest_path_tfcg
#SBATCH -p general
#SBATCH -o pipeline_shortest_path_tfcg_%j.txt
#SBATCH -e pipeline_shortest_path_tfcg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=58G
#SBATCH --time=36:00:00
errcho(){ >&2 echo $@; }
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
################################################################
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
errcho separate_claims
time python separate_claims.py
################################################################
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
errcho embeddings
time python embed.py -ft tfcg -mp roberta-base-nli-stsb-mean-tokens
################################################################
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
errcho create graphs
################################################################
errcho fetching
time python fetch_fred.py -ft tfcg -cpu 48
################################################################
errcho compiling
time python compile_fred.py -ft tfcg -cpu 48 
time python create_co_occur.py -ft tfcg_co
################################################################
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/process_graphs/
errcho calculate stats
time python calculate_stats.py -gc fred -gt tfcg
time python calculate_stats.py -gc co_occur -gt tfcg_co
time python compile_stats.py
################################################################
errcho leave1out
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_leave1out.py -fc fred -ft tfcg -cpu 48
time python create_leave1out.py -fc fred -ft tfcg_co -cpu 48
################################################################
errcho finding shortest paths tfcg
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
time python find_shortest_paths.py -st tfcg -ft ffcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -gt undirected
time python find_shortest_paths.py -st tfcg -ft tfcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -gt undirected
time python order_paths.py -fcg fred -ft tfcg
################################################################
errcho find baseline
time python find_baseline.py -bt knn -cpu 48 -mp roberta-base-nli-stsb-mean-tokens
time python find_baseline.py -bt all -mp roberta-base-nli-stsb-mean-tokens
################################################################
errcho plot graphs 
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/plot_graphs/
time python plot_sp.py -fcg fred -ft tfcg -pt roc
time python plot_sp.py -fcg fred -ft tfcg -pt dist