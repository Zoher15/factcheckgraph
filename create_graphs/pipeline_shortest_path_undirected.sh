#!/bin/bash
#SBATCH -J pipeline_shortest_path_undirected
#SBATCH -p general
#SBATCH -o pipeline_shortest_path_undirected_%j.txt
#SBATCH -e pipeline_shortest_path_undirected_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=58G
#SBATCH --time=36:00:00
errcho(){ >&2 echo $@; }
errcho create graphs
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_fred.py -ft tfcg -cpu 48 -p -gt undirected
time python create_fred.py -ft ffcg -cpu 48 -p -gt undirected
time python create_fred.py -ft tfcg -cpu 48 -cf 1 -gt undirected
time python create_fred.py -ft ffcg -cpu 48 -cf 1 -gt undirected
time python create_fred.py -ft ufcg -cf 1 -gt undirected
time python create_co_occur.py -ft tfcg_co
time python create_co_occur.py -ft ffcg_co
time python create_co_occur.py -ft ufcg_co
errcho find intersect
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/process_graphs/
# time python find_intersect.py -fcg fred -kg dbpedia
errcho calculate stats
time python calculate_stats.py -gc fred -gt tfcg
time python calculate_stats.py -gc fred -gt ffcg
time python calculate_stats.py -gc fred -gt ufcg
time python calculate_stats.py -gc co_occur -gt tfcg_co
time python calculate_stats.py -gc co_occur -gt ffcg_co
time python calculate_stats.py -gc co_occur -gt ufcg_co
time python compile_stats.py
################################################################
errcho leave1out
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_leave1out.py -fc fred -ft tfcg -cpu 48
# time python create_leave1out.py -fc co_occur -ft tfcg_co -cpu 48
################################################################
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
errcho embeddings
time python embed.py -ft ffcg -fc co_occur -gt undirected
time python embed.py -ft tfcg -fc co_occur -gt undirected
time python embed.py -ft ffcg -fc fred -gt undirected
time python embed.py -ft tfcg -fc fred -gt undirected
time python embed.py -ft ffcg -fc co_occur -mp roberta-base-nli-stsb-mean-tokens -gt undirected
time python embed.py -ft tfcg -fc co_occur -mp roberta-base-nli-stsb-mean-tokens -gt undirected
time python embed.py -ft ffcg -fc fred -mp roberta-base-nli-stsb-mean-tokens -gt undirected
time python embed.py -ft tfcg -fc fred -mp roberta-base-nli-stsb-mean-tokens -gt undirected
################################################################
# errcho finding shortest paths
# time python find_shortest_paths.py -ft ffcg -fc fred -cpu 48 -gt undirected
# time python find_shortest_paths.py -ft tfcg -fc fred -cpu 48 -gt undirected
errcho finding shortest paths 2
time python find_shortest_paths.py -ft ffcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -gt undirected
time python find_shortest_paths.py -ft tfcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -gt undirected
errcho plot graphs 
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/plot_graphs/
time python plot_sp.py -fcg fred -pt roc -gt undirected
time python plot_sp.py -fcg fred -pt dist -gt undirected