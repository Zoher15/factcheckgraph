#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=16gb,walltime=3:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N create_leave1out
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph/create_graphs/
time python create_leave1out.py -m false
time python create_leave1out.py -m true