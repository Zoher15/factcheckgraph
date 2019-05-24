#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=128gb,walltime=0:30:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N TFCG
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/TFCG
time klinker linkpred TFCG_uris.txt TFCG_edgelist.npy TFCG_dbpedia_triples.txt TFCG_degree_u.json -u -n 12
time klinker linkpred TFCG_uris.txt TFCG_edgelist.npy TFCG_dbpedia_triples.txt TFCG_logdegree_u.json -u -n 12 -w logdegree
time klinker linkpred TFCG_uris.txt TFCG_edgelist.npy TFCG_negative_dbpedia_triples.txt TFCG_negative_degree_u.json -u -n 12
time klinker linkpred TFCG_uris.txt TFCG_edgelist.npy TFCG_negative_dbpedia_triples.txt TFCG_negative_logdegree_u.json -u -n 12 -w logdegree