#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=180gb,walltime=3:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N DBPedia_KLinker
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\ Data/DBPedia\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\ Data/DBPedia\ Data/dbpedia_edgelist.npy DBPedia/Intersect_true_pairs_DBPedia_IDs.txt DBPedia/Intersect_true_pairs_DBPedia_IDs.json -u -n 12
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\ Data/DBPedia\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\ Data/DBPedia\ Data/dbpedia_edgelist.npy DBPedia/Intersect_false_pairs_DBPedia_IDs.txt DBPedia/Intersect_false_pairs_DBPedia_IDs.json -u -n 12