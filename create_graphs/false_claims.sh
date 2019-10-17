#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=64gb,walltime=36:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N FalseClaims
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/

time python fredlib.py false