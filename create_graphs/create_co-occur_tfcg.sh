#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=128gb,walltime=1:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N create_co-occur_tfcg
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph/create_graphs/
time python create_co-occur.py -ft tfcg_co