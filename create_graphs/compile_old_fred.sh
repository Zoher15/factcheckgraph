#!/bin/bash
#SBATCH -J compile_old_fred
#SBATCH -p general
#SBATCH -o compile_old_fred_%j.txt
#SBATCH -e compile_old_fred_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=3:00:00
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
#create old
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_old.py -fcg old_fred -ft tfcg_old
time python create_old.py -fcg old_fred -ft ffcg_old
time python create_old.py -fcg old_fred -ft ufcg_old
#find intersect
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/process_graphs/
time python find_intersect.py -fcg old_fred -kg dbpedia
#calculate stats
time python calculate_stats.py -gc old_fred -gt tfcg_old
time python calculate_stats.py -gc old_fred -gt ffcg_old
time python calculate_stats.py -gc old_fred -gt ufcg_old
time python compile_stats.py
#klinker
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/old_fred/
klinker linkpred tfcg_old/data/tfcg_old_nodes.txt tfcg_old/data/tfcg_old_edgelistID.npy tfcg_old/data/intersect_all_entityPairs_dbpedia_old_fred_tfcg_old_IDs_klformat.txt tfcg_old/data/intersect_all_entityPairs_dbpedia_old_fred_tfcg_old_IDs.json -u -n 48 -w logdegree
klinker linkpred ffcg_old/data/ffcg_old_nodes.txt ffcg_old/data/ffcg_old_edgelistID.npy ffcg_old/data/intersect_all_entityPairs_dbpedia_old_fred_ffcg_old_IDs_klformat.txt ffcg_old/data/intersect_all_entityPairs_dbpedia_old_fred_ffcg_old_IDs.json -u -n 48 -w logdegree
klinker linkpred ufcg_old/data/ufcg_old_nodes.txt ufcg_old/data/ufcg_old_edgelistID.npy ufcg_old/data/intersect_all_entityPairs_dbpedia_old_fred_ufcg_old_IDs_klformat.txt ufcg_old/data/intersect_all_entityPairs_dbpedia_old_fred_ufcg_old_IDs.json -u -n 48 -w logdegree
#find adj pairs
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
time python find_adj_pairs.py -fcg old_fred -kg dbpedia
#plot
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/plot_graphs/
time python plot_adj_pairs.py -fcg old_fred -kg dbpedia