#!/bin/bash

#SBATCH -J klinker_fred3
#SBATCH -p general
#SBATCH -o klinker_fred3_%j.txt
#SBATCH -e klinker_fred3_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/fred3/
klinker linkpred tfcg3/data/tfcg3_nodes.txt tfcg3/data/tfcg3_edgelistID.npy tfcg3/data/intersect_all_entityPairs_dbpedia_fred3_tfcg3_IDs_klformat.txt tfcg3/data/intersect_all_entityPairs_dbpedia_fred3_tfcg3_IDs.json -u -n 48 -w logdegree
klinker linkpred ffcg3/data/ffcg3_nodes.txt ffcg3/data/ffcg3_edgelistID.npy ffcg3/data/intersect_all_entityPairs_dbpedia_fred3_ffcg3_IDs_klformat.txt ffcg3/data/intersect_all_entityPairs_dbpedia_fred3_ffcg3_IDs.json -u -n 48 -w logdegree
klinker linkpred ufcg3/data/ufcg3_nodes.txt ufcg3/data/ufcg3_edgelistID.npy ufcg3/data/intersect_all_entityPairs_dbpedia_fred3_ufcg3_IDs_klformat.txt ufcg3/data/intersect_all_entityPairs_dbpedia_fred3_ufcg3_IDs.json -u -n 48 -w logdegree