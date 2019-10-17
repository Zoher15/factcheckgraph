#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=180gb,walltime=24:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N Process DBPedia
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/
time python process_graphs.py 