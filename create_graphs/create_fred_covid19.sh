#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=16gb,walltime=12:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N create_fred_covid19
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_covid19.py