
#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=5gb,walltime=2:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N 22_Corr_Co
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2
python experiment2.py 420000 440000 22
				