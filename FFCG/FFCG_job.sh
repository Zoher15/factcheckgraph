#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=128gb,walltime=0:30:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N FFCG
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph
time klinker linkpred FFCG_uris.txt FFCG_edgelist.npy FFCG_dbpedia_triples.txt FFCG_degree_u.json -u -n 12
time klinker linkpred FFCG_uris.txt FFCG_edgelist.npy FFCG_dbpedia_triples.txt FFCG_logdegree_u.json -u -n 12 -w logdegree
time klinker linkpred FFCG_uris.txt FFCG_edgelist.npy FFCG_negative_dbpedia_triples.txt FFCG_negative_degree_u.json -u -n 12
time klinker linkpred FFCG_uris.txt FFCG_edgelist.npy FFCG_negative_dbpedia_triples.txt FFCG_negative_logdegree_u.json -u -n 12 -w logdegree