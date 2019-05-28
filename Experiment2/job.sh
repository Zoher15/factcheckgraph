#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=255gb,walltime=2:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N Experiment2
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/DBPedia\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/DBPedia\ Data/dbpedia_edgelist.npy trueclaim_triples.txt trueclaim_degree_u.json -u -n 12
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/DBPedia\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/DBPedia\ Data/dbpedia_edgelist.npy trueclaim_triples.txt trueclaim_logdegree_u.json -u -n 12 -w logdegree
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/DBPedia\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/DBPedia\ Data/dbpedia_edgelist.npy falseclaim_triples.txt falseclaim_degree_u.json -u -n 12
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/DBPedia\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/DBPedia\ Data/dbpedia_edgelist.npy falseclaim_triples.txt falseclaim_logdegree_u.json -u -n 12 -w logdegree

# #PBS -k o
# #PBS -l nodes=1:ppn=1,vmem=64gb,walltime=2:00:00
# #PBS -M zoher.kachwala@gmail.com
# #PBS -m abe
# #PBS -N Experiment2
# #PBS -j oe
# source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
# conda activate
# cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2
# time python experiment2.py