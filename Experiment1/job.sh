#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=64gb,walltime=1:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N FactCheckGraph
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1
time python experiment1.py TFCG 1



# #PBS -k o
# #PBS -l nodes=1:ppn=12,vmem=128gb,walltime=0:05:00
# #PBS -M zoher.kachwala@gmail.com
# #PBS -m abe
# #PBS -N IntersectFCG
# #PBS -j oe
# source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
# conda activate env-kl
# cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/
# time klinker linkpred TFCG/TFCG_uris.txt TFCG/TFCG_edgelist.npy Intersect_triples_TFCG_IDs.txt Intersect_TFCG_logdegree_u.json -u -n 12 -w logdegree
# time klinker linkpred FFCG/FFCG_uris.txt FFCG/FFCG_edgelist.npy Intersect_triples_FFCG_IDs.txt Intersect_FFCG_logdegree_u.json -u -n 12 -w logdegree
# time klinker linkpred TFCG/TFCG_uris.txt TFCG/TFCG_edgelist.npy Random_intersect_triples_TFCG_IDs.txt Random_intersect_TFCG_logdegree_u.json -u -n 12 -w logdegree
# time klinker linkpred FFCG/FFCG_uris.txt FFCG/FFCG_edgelist.npy Random_intersect_triples_FFCG_IDs.txt Random_intersect_FFCG_logdegree_u.json -u -n 12 -w logdegree
# time klinker linkpred TFCG/TFCG_uris.txt TFCG/TFCG_edgelist.npy Random2_intersect_triples_TFCG_IDs.txt Random2_intersect_TFCG_logdegree_u.json -u -n 12 -w logdegree
# time klinker linkpred FFCG/FFCG_uris.txt FFCG/FFCG_edgelist.npy Random2_intersect_triples_FFCG_IDs.txt Random2_intersect_FFCG_logdegree_u.json -u -n 12 -w logdegree
