#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=128gb,walltime=0:05:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N IntersectFCG
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/
# # time klinker linkpred TFCG/TFCG_uris.txt TFCG/TFCG_edgelist.npy Intersect_triples_TFCG_IDs.txt TFCG/Intersect_TFCG.json -u -n 12 -w logdegree
# # time klinker linkpred FFCG/FFCG_uris.txt FFCG/FFCG_edgelist.npy Intersect_triples_FFCG_IDs.txt FFCG/Intersect_FFCG.json -u -n 12 -w logdegree
# time klinker linkpred TFCG/TFCG_uris.txt TFCG/TFCG_edgelist.npy Intersect_true_pairs_TFCG_IDs.txt TFCG/Intersect_true_pairs_TFCG_IDs.json -u -n 12 -w logdegree
# time klinker linkpred FFCG/FFCG_uris.txt FFCG/FFCG_edgelist.npy Intersect_true_pairs_FFCG_IDs.txt FFCG/Intersect_true_pairs_FFCG_IDs.json -u -n 12 -w logdegree
# time klinker linkpred FCG/FCG_uris.txt FCG/FCG_edgelist.npy Intersect_true_pairs_FCG_IDs.txt FCG/Intersect_true_pairs_FCG_IDs.json -u -n 12 -w logdegree
time klinker linkpred FCG_co/FCG_co_uris.txt FCG_co/FCG_co_edgelist.npy FCG/Intersect_true_pairs_FCG_IDs.txt FCG/Intersect_true_pairs_FCG_co_IDs.json -u -n 12 -w logdegree
# time klinker linkpred TFCG/TFCG_uris.txt TFCG/TFCG_edgelist.npy Intersect_false_pairs_TFCG_IDs.txt TFCG/Intersect_false_pairs_TFCG_IDs.json -u -n 12 -w logdegree
# time klinker linkpred FFCG/FFCG_uris.txt FFCG/FFCG_edgelist.npy Intersect_false_pairs_FFCG_IDs.txt FFCG/Intersect_false_pairs_FFCG_IDs.json -u -n 12 -w logdegree
# time klinker linkpred FCG/FCG_uris.txt FCG/FCG_edgelist.npy Intersect_false_pairs_FCG_IDs.txt FCG/Intersect_false_pairs_FCG_IDs.json -u -n 12 -w logdegree
time klinker linkpred FCG_co/FCG_co_uris.txt FCG_co/FCG_co_edgelist.npy FCG_co/Intersect_false_pairs_FCG_IDs.txt FCG_co/Intersect_false_pairs_FCG_co_IDs.json -u -n 12 -w logdegree
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