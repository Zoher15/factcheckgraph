#!/bin/bash
#SBATCH -J pipeline_shortest_path_directed
#SBATCH -p general
#SBATCH -o pipeline_shortest_path_directed_%j.txt
#SBATCH -e pipeline_shortest_path_directed_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=58G
#SBATCH --time=6:00:00
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_fred.py -ft tfcg -cpu 48 -p -gt directed
time python create_fred.py -ft ffcg -cpu 48 -p -gt directed
time python create_fred.py -ft tfcg -cpu 48 -cf 1 -gt directed
time python create_fred.py -ft ffcg -cpu 48 -cf 1 -gt directed
time python create_fred.py -ft ufcg -cf 1 -gt directed
time python calculate_stats.py -gc co-occur -gt tfcg_co
time python calculate_stats.py -gc co-occur -gt ffcg_co
time python calculate_stats.py -gc co-occur -gt ufcg_co
# find intersect
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/process_graphs/
time python find_intersect.py -fcg fred1 -kg dbpedia
#calculate stats
time python calculate_stats.py -gc fred -gt tfcg
time python calculate_stats.py -gc fred -gt ffcg
time python calculate_stats.py -gc fred -gt ufcg
time python calculate_stats.py -gc co-occur -gt tfcg_co
time python calculate_stats.py -gc co-occur -gt ffcg_co
time python calculate_stats.py -gc co-occur -gt ufcg_co
time python compile_stats.py
################################################################
# leave1out
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_leave1out.py -fc fred -ft tfcg -cpu 48
time python create_leave1out.py -fc co_occur -ft tfcg_co -cpu 48
################################################################
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
#embeddings
# time python embed.py -ft ffcg -fc co_occur -gt directed
# time python embed.py -ft tfcg -fc co_occur -gt directed
# time python embed.py -ft ffcg -fc fred -gt directed
# time python embed.py -ft tfcg -fc fred -gt directed
time python embed.py -ft ffcg -fc co_occur -mp roberta-base-nli-stsb-mean-tokens -gt directed
time python embed.py -ft tfcg -fc co_occur -mp roberta-base-nli-stsb-mean-tokens -gt directed
time python embed.py -ft ffcg -fc fred -mp roberta-base-nli-stsb-mean-tokens -gt directed
time python embed.py -ft tfcg -fc fred -mp roberta-base-nli-stsb-mean-tokens -gt directed
################################################################
#finding shortest paths
# time python find_shortest_paths.py -ft ffcg -fc fred -cpu 48 -gt directed
# time python find_shortest_paths.py -ft tfcg -fc fred -cpu 48 -gt directed
#finding shortest paths 2
time python find_shortest_paths.py -ft ffcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -gt directed
time python find_shortest_paths.py -ft tfcg -mp roberta-base-nli-stsb-mean-tokens -fc fred -cpu 48 -gt directed