#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=128gb,walltime=1:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N IntersectFCG_co
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/
# time klinker linkpred TFCG/TFCG_uris.txt TFCG/TFCG_edgelist.npy Intersect_triples_TFCG_IDs.txt Intersect_TFCG.json -u -n 12 -w logdegree
# time klinker linkpred FFCG/FFCG_uris.txt FFCG/FFCG_edgelist.npy Intersect_triples_FFCG_IDs.txt Intersect_FFCG.json -u -n 12 -w logdegree
time klinker linkpred TFCG_co/TFCG_co_uris.txt TFCG_co/TFCG_co_edgelist.npy TFCG_co/Intersect_true_pairs_TFCG_co_IDs.txt TFCG_co/Intersect_true_pairs_TFCG_co_IDs.json -u -n 12 -w logdegree
time klinker linkpred FFCG_co/FFCG_co_uris.txt FFCG_co/FFCG_co_edgelist.npy FFCG_co/Intersect_true_pairs_FFCG_co_IDs.txt FFCG_co/Intersect_true_pairs_FFCG_co_IDs.json -u -n 12 -w logdegree
time klinker linkpred TFCG_co/TFCG_co_uris.txt TFCG_co/TFCG_co_edgelist.npy TFCG_co/Intersect_false_pairs_TFCG_co_IDs.txt TFCG_co/Intersect_false_pairs_TFCG_co_IDs.json -u -n 12 -w logdegree
time klinker linkpred FFCG_co/FFCG_co_uris.txt FFCG_co/FFCG_co_edgelist.npy FFCG_co/Intersect_false_pairs_FFCG_co_IDs.txt FFCG_co/Intersect_false_pairs_FFCG_co_IDs.json -u -n 12 -w logdegree
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

