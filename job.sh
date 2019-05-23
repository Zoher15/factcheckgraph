#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=128gb,walltime=2:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N FactCheckGraph
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph
time python dbpedia_neo4jscript.py TFCG 1
time python dbpedia_neo4jscript.py FFCG 1