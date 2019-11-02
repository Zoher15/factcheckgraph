#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=64gb,walltime=0:10:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N plot_graphs
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph/plot_graphs/
# time python plot_adj_pairs.py -fcg fred -kg dbpedia 
# time python plot_adj_pairs.py -fcg backbone_df -kg dbpedia 
# time python plot_adj_pairs.py -fcg largest_ccf -kg dbpedia 
time python plot_adj_pairs.py -fcg old_fred -kg dbpedia 