#!/bin/bash

#SBATCH -J klinker_bbdf
#SBATCH -p general
#SBATCH -o klinker_bbdf_%j.txt
#SBATCH -e klinker_bbdf_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=05:00:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/backbone_df/
klinker linkpred tfcg_bbdf/data/tfcg_bbdf_nodes.txt tfcg_bbdf/data/tfcg_bbdf_edgelistID.npy tfcg_bbdf/data/intersect_all_entityPairs_dbpedia_backbone_df_tfcg_bbdf_IDs_klformat.txt tfcg_bbdf/data/intersect_all_entityPairs_dbpedia_backbone_df_tfcg_bbdf_IDs.json -u -n 20 -w logdegree
klinker linkpred ffcg_bbdf/data/ffcg_bbdf_nodes.txt ffcg_bbdf/data/ffcg_bbdf_edgelistID.npy ffcg_bbdf/data/intersect_all_entityPairs_dbpedia_backbone_df_ffcg_bbdf_IDs_klformat.txt ffcg_bbdf/data/intersect_all_entityPairs_dbpedia_backbone_df_ffcg_bbdf_IDs.json -u -n 20 -w logdegree
klinker linkpred ufcg_bbdf/data/ufcg_bbdf_nodes.txt ufcg_bbdf/data/ufcg_bbdf_edgelistID.npy ufcg_bbdf/data/intersect_all_entityPairs_dbpedia_backbone_df_ufcg_bbdf_IDs_klformat.txt ufcg_bbdf/data/intersect_all_entityPairs_dbpedia_backbone_df_ufcg_bbdf_IDs.json -u -n 20 -w logdegree