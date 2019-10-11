#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=180gb,walltime=12:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N IntersectFCG
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/

time klinker linkpred FCG_ct/FCG_ct_nodes.txt FCG_ct/FCG_ct_edgelistID.npy FCG_ct/Intersect_all_entityPairs_FCG_ct_IDs_klformat.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\ Data/FCG\ Data/Experiment1/FCG_ct/Intersect_all_entityPairs_FCG_ct_IDs.json -u -n 12 -w logdegree
time klinker linkpred FCG_ct/FCG_ct_nodes.txt FCG_ct/FCG_ct_edgelistID.npy FCG_ct/Intersect_true_entityPairs_FCG_ct_IDs_klformat.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\ Data/FCG\ Data/Experiment1/FCG_ct/Intersect_true_entityPairs_FCG_ct_IDs.json -u -n 12 -w logdegree
time klinker linkpred TFCG_ct/TFCG_ct_nodes.txt TFCG_ct/TFCG_ct_edgelistID.npy TFCG_ct/Intersect_all_entityPairs_TFCG_ct_IDs_klformat.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\ Data/FCG\ Data/Experiment1/TFCG_ct/Intersect_all_entityPairs_TFCG_ct_IDs.json -u -n 12 -w logdegree
time klinker linkpred TFCG_ct/TFCG_ct_nodes.txt TFCG_ct/TFCG_ct_edgelistID.npy TFCG_ct/Intersect_true_entityPairs_TFCG_ct_IDs_klformat.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\ Data/FCG\ Data/Experiment1/TFCG_ct/Intersect_true_entityPairs_TFCG_ct_IDs.json -u -n 12 -w logdegree
time klinker linkpred FFCG_ct/FFCG_ct_nodes.txt FFCG_ct/FFCG_ct_edgelistID.npy FFCG_ct/Intersect_all_entityPairs_FFCG_ct_IDs_klformat.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\ Data/FCG\ Data/Experiment1/FFCG_ct/Intersect_all_entityPairs_FFCG_ct_IDs.json -u -n 12 -w logdegree
time klinker linkpred FFCG_ct/FFCG_ct_nodes.txt FFCG_ct/FFCG_ct_edgelistID.npy FFCG_ct/Intersect_true_entityPairs_FFCG_ct_IDs_klformat.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\ Data/FCG\ Data/Experiment1/FFCG_ct/Intersect_true_entityPairs_FFCG_ct_IDs.json -u -n 12 -w logdegree
# time klinker linkpred FCG/FCG_uris.txt FCG/FCG_edgelist.npy FCG/Intersect_all_pairs_FCG_IDs.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\ Data/FCG\ Data/Experiment1/FCG/Intersect_FCG.json -u -n 12 -w logdegree
# time klinker linkpred FCG_co/FCG_co_uris.txt FCG_co/FCG_co_edgelist.npy FCG_co/Intersect_all_pairs_FCG_co_IDs.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\ Data/FCG\ Data/Experiment1/FCG_co/Intersect_FCG_co.json -u -n 12 -w logdegree

# # time klinker linkpred TFCG/TFCG_uris.txt TFCG/TFCG_edgelist.npy Intersect_triples_TFCG_IDs.txt TFCG/Intersect_TFCG.json -u -n 12 -w logdegree
# # time klinker linkpred FFCG/FFCG_uris.txt FFCG/FFCG_edgelist.npy Intersect_triples_FFCG_IDs.txt FFCG/Intersect_FFCG.json -u -n 12 -w logdegree
# time klinker linkpred TFCG/TFCG_uris.txt TFCG/TFCG_edgelist.npy Intersect_true_pairs_TFCG_IDs.txt TFCG/Intersect_true_pairs_TFCG_IDs.json -u -n 12 -w logdegree
# time klinker linkpred FFCG/FFCG_uris.txt FFCG/FFCG_edgelist.npy Intersect_true_pairs_FFCG_IDs.txt FFCG/Intersect_true_pairs_FFCG_IDs.json -u -n 12 -w logdegree
# time klinker linkpred FCG/FCG_uris.txt FCG/FCG_edgelist.npy Intersect_true_pairs_FCG_IDs.txt FCG/Intersect_true_pairs_FCG_IDs.json -u -n 12 -w logdegree
# time klinker linkpred FCG_co/FCG_co_uris.txt FCG_co/FCG_co_edgelist.npy FCG_co/Intersect_true_pairs_FCG_co_IDs.txt FCG_co/Intersect_true_pairs_FCG_co_IDs.json -u -n 1 -w logdegree
# time klinker linkpred TFCG/TFCG_uris.txt TFCG/TFCG_edgelist.npy Intersect_false_pairs_TFCG_IDs.txt TFCG/Intersect_false_pairs_TFCG_IDs.json -u -n 12 -w logdegree
# time klinker linkpred FFCG/FFCG_uris.txt FFCG/FFCG_edgelist.npy Intersect_false_pairs_FFCG_IDs.txt FFCG/Intersect_false_pairs_FFCG_IDs.json -u -n 12 -w logdegree
# time klinker linkpred FCG/FCG_uris.txt FCG/FCG_edgelist.npy Intersect_false_pairs_FCG_IDs.txt FCG/Intersect_false_pairs_FCG_IDs.json -u -n 12 -w logdegree
# time klinker linkpred FCG_co/FCG_co_uris.txt FCG_co/FCG_co_edgelist.npy FCG_co/Intersect_false_pairs_FCG_co_IDs.txt FCG_co/Intersect_false_pairs_FCG_co_IDs.json -u -n 1 -w logdegree
# time klinker linkpred TFCG/TFCG_uris.txt TFCG/TFCG_edgelist.npy Intersect_false_pairs2_TFCG_IDs.txt TFCG/Intersect_false_pairs2_TFCG_IDs.json -u -n 12 -w logdegree
# time klinker linkpred FFCG/FFCG_uris.txt FFCG/FFCG_edgelist.npy Intersect_false_pairs2_FFCG_IDs.txt FFCG/Intersect_false_pairs2_FFCG_IDs.json -u -n 12 -w logdegree
# # #PBS -k o
# #PBS -l nodes=1:ppn=1,vmem=64gb,walltime=1:00:00
# #PBS -M zoher.kachwala@gmail.com
# #PBS -m abe
# #PBS -N FactCheckGraph
# #PBS -j oe
# source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
# conda activate
# cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1
# time python experiment1.py TFCG 1




# #PBS -k o
# #PBS -l nodes=1:ppn=1,vmem=50gb,walltime=1:00:00
# #PBS -M zoher.kachwala@gmail.com
# #PBS -m abe
# #PBS -N 1_Corr
# #PBS -j oe
# source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
# conda activate
# cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1
# python experiment1.py TFCG 1 0 562330