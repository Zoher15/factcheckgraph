#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=64gb,walltime=3:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N klinker_co-occur
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/co-occur/
klinker linkpred tfcg_co/data/tfcg_co_nodes.txt tfcg_co/data/tfcg_co_edgelistID.npy tfcg_co/data/intersect_all_entityPairs_dbpedia_fred_tfcg_co_IDs_klformat.txt tfcg_co/data/intersect_all_entityPairs_dbpedia_fred_tfcg_co_IDs.json -u -n 12 -w logdegree
klinker linkpred ffcg_co/data/ffcg_co_nodes.txt ffcg_co/data/ffcg_co_edgelistID.npy ffcg_co/data/intersect_all_entityPairs_dbpedia_fred_ffcg_co_IDs_klformat.txt ffcg_co/data/intersect_all_entityPairs_dbpedia_fred_ffcg_co_IDs.json -u -n 12 -w logdegree
klinker linkpred ufcg_co/data/ufcg_co_nodes.txt ufcg_co/data/ufcg_co_edgelistID.npy ufcg_co/data/intersect_all_entityPairs_dbpedia_fred_ufcg_co_IDs_klformat.txt ufcg_co/data/intersect_all_entityPairs_dbpedia_fred_ufcg_co_IDs.json -u -n 12 -w logdegree