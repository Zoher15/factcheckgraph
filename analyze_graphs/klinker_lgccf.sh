#!/bin/bash

#SBATCH -J klinker_lgccf
#SBATCH -p general
#SBATCH -o klinker_lgccf_%j.txt
#SBATCH -e klinker_lgccf_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=05:00:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/largest_ccf/
klinker linkpred tfcg_lgccf/data/tfcg_lgccf_nodes.txt tfcg_lgccf/data/tfcg_lgccf_edgelistID.npy tfcg_lgccf/data/intersect_all_entityPairs_dbpedia_largest_ccf_tfcg_lgccf_IDs_klformat.txt tfcg_lgccf/data/intersect_all_entityPairs_dbpedia_largest_ccf_tfcg_lgccf_IDs.json -u -n 20 -w logdegree
klinker linkpred ffcg_lgccf/data/ffcg_lgccf_nodes.txt ffcg_lgccf/data/ffcg_lgccf_edgelistID.npy ffcg_lgccf/data/intersect_all_entityPairs_dbpedia_largest_ccf_ffcg_lgccf_IDs_klformat.txt ffcg_lgccf/data/intersect_all_entityPairs_dbpedia_largest_ccf_ffcg_lgccf_IDs.json -u -n 20 -w logdegree
klinker linkpred ufcg_lgccf/data/ufcg_lgccf_nodes.txt ufcg_lgccf/data/ufcg_lgccf_edgelistID.npy ufcg_lgccf/data/intersect_all_entityPairs_dbpedia_largest_ccf_ufcg_lgccf_IDs_klformat.txt ufcg_lgccf/data/intersect_all_entityPairs_dbpedia_largest_ccf_ufcg_lgccf_IDs.json -u -n 20 -w logdegree