#!/bin/bash

#SBATCH -J klinker_fred1
#SBATCH -p general
#SBATCH -o klinker_fred1_%j.txt
#SBATCH -e klinker_fred1_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/fred1/
klinker linkpred tfcg1/data/tfcg1_nodes.txt tfcg1/data/tfcg1_edgelistID.npy tfcg1/data/intersect_all_entityPairs_dbpedia_fred1_tfcg1_IDs_klformat.txt tfcg1/data/intersect_all_entityPairs_dbpedia_fred1_tfcg1_IDs.json -u -n 48 -w logdegree
klinker linkpred ffcg1/data/ffcg1_nodes.txt ffcg1/data/ffcg1_edgelistID.npy ffcg1/data/intersect_all_entityPairs_dbpedia_fred1_ffcg1_IDs_klformat.txt ffcg1/data/intersect_all_entityPairs_dbpedia_fred1_ffcg1_IDs.json -u -n 48 -w logdegree
klinker linkpred ufcg1/data/ufcg1_nodes.txt ufcg1/data/ufcg1_edgelistID.npy ufcg1/data/intersect_all_entityPairs_dbpedia_fred1_ufcg1_IDs_klformat.txt ufcg1/data/intersect_all_entityPairs_dbpedia_fred1_ufcg1_IDs.json -u -n 48 -w logdegree