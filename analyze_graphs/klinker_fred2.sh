#!/bin/bash

#SBATCH -J klinker_fred2
#SBATCH -p general
#SBATCH -o klinker_fred2_%j.txt
#SBATCH -e klinker_fred2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/fred2/
klinker linkpred tfcg2/data/tfcg2_nodes.txt tfcg2/data/tfcg2_edgelistID.npy tfcg2/data/intersect_all_entityPairs_dbpedia_fred2_tfcg2_IDs_klformat.txt tfcg2/data/intersect_all_entityPairs_dbpedia_fred2_tfcg2_IDs.json -u -n 48 -w logdegree
klinker linkpred ffcg2/data/ffcg2_nodes.txt ffcg2/data/ffcg2_edgelistID.npy ffcg2/data/intersect_all_entityPairs_dbpedia_fred2_ffcg2_IDs_klformat.txt ffcg2/data/intersect_all_entityPairs_dbpedia_fred2_ffcg2_IDs.json -u -n 48 -w logdegree
klinker linkpred ufcg2/data/ufcg2_nodes.txt ufcg2/data/ufcg2_edgelistID.npy ufcg2/data/intersect_all_entityPairs_dbpedia_fred2_ufcg2_IDs_klformat.txt ufcg2/data/intersect_all_entityPairs_dbpedia_fred2_ufcg2_IDs.json -u -n 48 -w logdegree