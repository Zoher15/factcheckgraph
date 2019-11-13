#!/bin/bash

#SBATCH -J klinker_fred
#SBATCH -p general
#SBATCH -o klinker_fred_%j.txt
#SBATCH -e klinker_fred_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=05:00:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/fred/
klinker linkpred tfcg/data/tfcg_nodes.txt tfcg/data/tfcg_edgelistID.npy tfcg/data/intersect_all_entityPairs_dbpedia_fred_tfcg_IDs_klformat.txt tfcg/data/intersect_all_entityPairs_dbpedia_fred_tfcg_IDs.json -u -n 20 -w logdegree
klinker linkpred ffcg/data/ffcg_nodes.txt ffcg/data/ffcg_edgelistID.npy ffcg/data/intersect_all_entityPairs_dbpedia_fred_ffcg_IDs_klformat.txt ffcg/data/intersect_all_entityPairs_dbpedia_fred_ffcg_IDs.json -u -n 20 -w logdegree
klinker linkpred ufcg/data/ufcg_nodes.txt ufcg/data/ufcg_edgelistID.npy ufcg/data/intersect_all_entityPairs_dbpedia_fred_ufcg_IDs_klformat.txt ufcg/data/intersect_all_entityPairs_dbpedia_fred_ufcg_IDs.json -u -n 20 -w logdegree