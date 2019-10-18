#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=180gb,walltime=24:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N process_dbpedia
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph/process_graphs/
time python save_graph_data.py -gc kg -gt dbpedia
time python calculate_stats.py -gc kg -gt dbpedia