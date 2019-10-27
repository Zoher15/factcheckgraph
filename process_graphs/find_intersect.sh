#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=64gb,walltime=0:5:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N find_intersect
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph/process_graphs/
time python find_intersect.py -fcg fred -kg dbpedia
time python find_intersect.py -fcg co-occur -kg dbpedia