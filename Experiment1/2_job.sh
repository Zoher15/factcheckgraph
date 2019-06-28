#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=180gb,walltime=24:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N 2_KLinker
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/DBPedia\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/DBPedia\ Data/dbpedia_edgelist.npy 2_Comb_triples_DBPedia_IDs.txt 2_Comb_triples_DBPedia_IDs.json -u -n 12