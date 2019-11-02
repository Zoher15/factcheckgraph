#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=64gb,walltime=1:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N klinker_lgccf
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/largest_ccf/
# klinker linkpred tfcg_lgccf/data/tfcg_lgccf_nodes.txt tfcg_lgccf/data/tfcg_lgccf_edgelistID.npy tfcg_lgccf/data/intersect_all_entityPairs_dbpedia_largest_ccf_tfcg_lgccf_IDs_klformat.txt tfcg_lgccf/data/intersect_all_entityPairs_dbpedia_largest_ccf_tfcg_lgccf_IDs.json -u -n 12 -w logdegree
# klinker linkpred ffcg_lgccf/data/ffcg_lgccf_nodes.txt ffcg_lgccf/data/ffcg_lgccf_edgelistID.npy ffcg_lgccf/data/intersect_all_entityPairs_dbpedia_largest_ccf_ffcg_lgccf_IDs_klformat.txt ffcg_lgccf/data/intersect_all_entityPairs_dbpedia_largest_ccf_ffcg_lgccf_IDs.json -u -n 12 -w logdegree
klinker linkpred ufcg_lgccf/data/ufcg_lgccf_nodes.txt ufcg_lgccf/data/ufcg_lgccf_edgelistID.npy ufcg_lgccf/data/intersect_all_entityPairs_dbpedia_largest_ccf_ufcg_lgccf_IDs_klformat.txt ufcg_lgccf/data/intersect_all_entityPairs_dbpedia_largest_ccf_ufcg_lgccf_IDs.json -u -n 12 -w logdegree