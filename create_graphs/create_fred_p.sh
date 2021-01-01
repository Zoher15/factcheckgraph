#!/bin/bash
#SBATCH -J create_fred_p
#SBATCH -p general
#SBATCH -o create_fred_p_%j.txt
#SBATCH -e create_fred_p_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python fetch_fred.py -ft tfcg -cpu 48 -p
time python fetch_fred.py -ft ffcg -cpu 48 -p 
time python compile_fred.py -ft tfcg -cpu 48 
time python compile_fred.py -ft ffcg -cpu 48
time python compile_fred.py -ft ufcg
################################################################
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/process_graphs/
# errcho find intersect
# time python find_intersect.py -fcg fred -kg dbpedia
################################################################
errcho calculate stats
time python calculate_stats.py -gc fred -gt tfcg
time python calculate_stats.py -gc fred -gt ffcg
time python calculate_stats.py -gc fred -gt ufcg
time python compile_stats.py
# time python create_fred.py -ft ufcg -cf 1 -gt undirected
# time python create_fred.py -ft ufcg
# time python create_backbone.py -fcg fred -ft tfcg -kg dbpedia
# time python create_backbone.py -fcg fred -ft ffcg -kg dbpedia
# time python create_backbone.py -fcg fred -ft ufcg -kg dbpedia
# time python create_largest_cc.py -fcg fred -ft tfcg 
# time python create_largest_cc.py -fcg fred -ft ffcg
# time python create_largest_cc.py -fcg fred -ft ufcg 

