#!/bin/bash
#SBATCH -J compile_fred1
#SBATCH -p general
#SBATCH -o compile_fred1_%j.txt
#SBATCH -e compile_fred1_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=4:00:00
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_fred.py -ft tfcg -cpu 48 -p -gt undirected
time python create_fred.py -ft ffcg -cpu 48 -p -gt undirected
time python create_fred.py -ft tfcg -cpu 48 -cf 1 -gt undirected
time python create_fred.py -ft ffcg -cpu 48 -cf 1 -gt undirected
time python create_fred.py -ft ufcg -cf 1 -gt undirected
# find intersect
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/process_graphs/
# time python find_intersect.py -fcg fred1 -kg dbpedia
#calculate stats
time python calculate_stats.py -gc fred -gt tfcg
time python calculate_stats.py -gc fred -gt ffcg
time python calculate_stats.py -gc fred -gt ufcg
time python compile_stats.py
#klinker
# conda activate env-kl
# cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/fred1/
# klinker linkpred tfcg1/data/tfcg1_nodes.txt tfcg1/data/tfcg1_edgelistID.npy tfcg1/data/intersect_all_entityPairs_dbpedia_fred1_tfcg1_IDs_klformat.txt tfcg1/data/intersect_all_entityPairs_dbpedia_fred1_tfcg1_IDs.json -u -n 48 -w logdegree
# klinker linkpred ffcg1/data/ffcg1_nodes.txt ffcg1/data/ffcg1_edgelistID.npy ffcg1/data/intersect_all_entityPairs_dbpedia_fred1_ffcg1_IDs_klformat.txt ffcg1/data/intersect_all_entityPairs_dbpedia_fred1_ffcg1_IDs.json -u -n 48 -w logdegree
# klinker linkpred ufcg1/data/ufcg1_nodes.txt ufcg1/data/ufcg1_edgelistID.npy ufcg1/data/intersect_all_entityPairs_dbpedia_fred1_ufcg1_IDs_klformat.txt ufcg1/data/intersect_all_entityPairs_dbpedia_fred1_ufcg1_IDs.json -u -n 48 -w logdegree
# #find adj pairs
# conda activate
# cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
# time python find_adj_pairs.py -fcg fred1 -kg dbpedia
# #plot
# cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/plot_graphs/
# time python plot_adj_pairs.py -fcg fred1 -kg dbpedia