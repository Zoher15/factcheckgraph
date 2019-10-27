#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=64gb,walltime=0:15:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N analyze_dbpedia
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph/analyze_graphs/
time python find_adj_pairs.py -fcg fred -kg dbpedia 
# time python calculate_stats.py -gc kg -gt dbpedia