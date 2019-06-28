#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=180gb,walltime=200:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N 1_KLinker
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/DBPedia\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/DBPedia\ Data/dbpedia_edgelist.npy 1_Comb_triples_DBPedia_IDs.txt 1_Comb_triples_DBPedia_IDs.json -u -n 12