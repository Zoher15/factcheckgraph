#!/bin/bash
#SBATCH -J compile_fred2
#SBATCH -p general
#SBATCH -o compile_fred2_%j.txt
#SBATCH -e compile_fred2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=7:00:00
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_fred.py -ft tfcg -cpu 48 -cf 2
time python create_fred.py -ft ffcg -cpu 48 -cf 2
time python create_fred.py -ft ufcg -cf 2
#find intersect
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/process_graphs/
time python find_intersect.py -fcg fred2 -kg dbpedia
#calculate stats
time python calculate_stats.py -gc fred2 -gt tfcg2
time python calculate_stats.py -gc fred2 -gt ffcg2
time python calculate_stats.py -gc fred2 -gt ufcg2
time python compile_stats.py
#klinker
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/fred2/
klinker linkpred tfcg2/data/tfcg2_nodes.txt tfcg2/data/tfcg2_edgelistID.npy tfcg2/data/intersect_all_entityPairs_dbpedia_fred2_tfcg2_IDs_klformat.txt tfcg2/data/intersect_all_entityPairs_dbpedia_fred2_tfcg2_IDs.json -u -n 48 -w logdegree
klinker linkpred ffcg2/data/ffcg2_nodes.txt ffcg2/data/ffcg2_edgelistID.npy ffcg2/data/intersect_all_entityPairs_dbpedia_fred2_ffcg2_IDs_klformat.txt ffcg2/data/intersect_all_entityPairs_dbpedia_fred2_ffcg2_IDs.json -u -n 48 -w logdegree
klinker linkpred ufcg2/data/ufcg2_nodes.txt ufcg2/data/ufcg2_edgelistID.npy ufcg2/data/intersect_all_entityPairs_dbpedia_fred2_ufcg2_IDs_klformat.txt ufcg2/data/intersect_all_entityPairs_dbpedia_fred2_ufcg2_IDs.json -u -n 48 -w logdegree
#find adj pairs
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
time python find_adj_pairs.py -fcg fred2 -kg dbpedia
#plot
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/plot_graphs/
time python plot_adj_pairs.py -fcg fred2 -kg dbpedia