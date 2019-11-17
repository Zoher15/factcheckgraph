#!/bin/bash
#SBATCH -J compile_fred3
#SBATCH -p general
#SBATCH -o compile_fred3_%j.txt
#SBATCH -e compile_fred3_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=4:00:00
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_fred.py -ft tfcg -cf 3
time python create_fred.py -ft ffcg -cf 3
time python create_fred.py -ft ufcg -cf 3
#find intersect
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/process_graphs/
time python find_intersect.py -fcg fred3 -kg dbpedia
#calculate stats
time python calculate_stats.py -gc fred3 -gt tfcg3
time python calculate_stats.py -gc fred3 -gt ffcg3
time python calculate_stats.py -gc fred3 -gt ufcg3
time python compile_stats.py
#klinker
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/fred3/
klinker linkpred tfcg3/data/tfcg3_nodes.txt tfcg3/data/tfcg3_edgelistID.npy tfcg3/data/intersect_all_entityPairs_dbpedia_fred3_tfcg3_IDs_klformat.txt tfcg3/data/intersect_all_entityPairs_dbpedia_fred3_tfcg3_IDs.json -u -n 48 -w logdegree
klinker linkpred ffcg3/data/ffcg3_nodes.txt ffcg3/data/ffcg3_edgelistID.npy ffcg3/data/intersect_all_entityPairs_dbpedia_fred3_ffcg3_IDs_klformat.txt ffcg3/data/intersect_all_entityPairs_dbpedia_fred3_ffcg3_IDs.json -u -n 48 -w logdegree
klinker linkpred ufcg3/data/ufcg3_nodes.txt ufcg3/data/ufcg3_edgelistID.npy ufcg3/data/intersect_all_entityPairs_dbpedia_fred3_ufcg3_IDs_klformat.txt ufcg3/data/intersect_all_entityPairs_dbpedia_fred3_ufcg3_IDs.json -u -n 48 -w logdegree
#find adj pairs
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
time python find_adj_pairs.py -fcg fred3 -kg dbpedia
#plot
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/plot_graphs/
time python plot_adj_pairs.py -fcg fred3 -kg dbpedia 