#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=32gb,walltime=0:45:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N DBpedia FCG
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/
time python dbpedia_script.py
