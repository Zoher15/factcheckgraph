#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=64gb,walltime=3:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N klinker_bbdf
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/backbone_df/
klinker linkpred tfcg_bbdf/data/tfcg_bbdf_nodes.txt tfcg_bbdf/data/tfcg_bbdf_edgelistID.npy tfcg_bbdf/data/intersect_all_entityPairs_dbpedia_fred_tfcg_bbdf_IDs_klformat.txt tfcg_bbdf/data/intersect_all_entityPairs_dbpedia_fred_tfcg_bbdf_IDs.json -u -n 12 -w logdegree
klinker linkpred ffcg_bbdf/data/ffcg_bbdf_nodes.txt ffcg_bbdf/data/ffcg_bbdf_edgelistID.npy ffcg_bbdf/data/intersect_all_entityPairs_dbpedia_fred_ffcg_bbdf_IDs_klformat.txt ffcg_bbdf/data/intersect_all_entityPairs_dbpedia_fred_ffcg_bbdf_IDs.json -u -n 12 -w logdegree
klinker linkpred ufcg_bbdf/data/ufcg_bbdf_nodes.txt ufcg_bbdf/data/ufcg_bbdf_edgelistID.npy ufcg_bbdf/data/intersect_all_entityPairs_dbpedia_fred_ufcg_bbdf_IDs_klformat.txt ufcg_bbdf/data/intersect_all_entityPairs_dbpedia_fred_ufcg_bbdf_IDs.json -u -n 12 -w logdegree