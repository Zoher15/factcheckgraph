#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=64gb,walltime=3:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N klinker_fred
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/fred/
klinker linkpred tfcg/data/tfcg_nodes.txt tfcg/data/tfcg_edgelistID.npy tfcg/data/intersect_all_entityPairs_dbpedia_fred_tfcg_IDs_klformat.txt tfcg/data/intersect_all_entityPairs_dbpedia_fred_tfcg_IDs.json -u -n 12 -w logdegree
klinker linkpred ffcg/data/ffcg_nodes.txt ffcg/data/ffcg_edgelistID.npy ffcg/data/intersect_all_entityPairs_dbpedia_fred_ffcg_IDs_klformat.txt ffcg/data/intersect_all_entityPairs_dbpedia_fred_ffcg_IDs.json -u -n 12 -w logdegree
klinker linkpred ufcg/data/ufcg_nodes.txt ufcg/data/ufcg_edgelistID.npy ufcg/data/intersect_all_entityPairs_dbpedia_fred_ufcg_IDs_klformat.txt ufcg/data/intersect_all_entityPairs_dbpedia_fred_ufcg_IDs.json -u -n 12 -w logdegree