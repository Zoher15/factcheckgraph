#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=64gb,walltime=3:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N klinker_old_fred
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/old_fred/
klinker linkpred tfcg_old/data/tfcg_old_nodes.txt tfcg_old/data/tfcg_old_edgelistID.npy tfcg_old/data/intersect_all_entityPairs_dbpedia_old_fred_tfcg_old_IDs_klformat.txt tfcg_old/data/intersect_all_entityPairs_dbpedia_old_fred_tfcg_old_IDs.json -u -n 12 -w logdegree
klinker linkpred ffcg_old/data/ffcg_old_nodes.txt ffcg_old/data/ffcg_old_edgelistID.npy ffcg_old/data/intersect_all_entityPairs_dbpedia_old_fred_ffcg_old_IDs_klformat.txt ffcg_old/data/intersect_all_entityPairs_dbpedia_old_fred_ffcg_old_IDs.json -u -n 12 -w logdegree
klinker linkpred ufcg_old/data/ufcg_old_nodes.txt ufcg_old/data/ufcg_old_edgelistID.npy ufcg_old/data/intersect_all_entityPairs_dbpedia_old_fred_ufcg_old_IDs_klformat.txt ufcg_old/data/intersect_all_entityPairs_dbpedia_old_fred_ufcg_old_IDs.json -u -n 12 -w logdegree