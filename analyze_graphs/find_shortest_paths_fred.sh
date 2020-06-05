#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=64gb,walltime=12:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N find_shortest_paths_fred
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
time python find_shortest_paths.py -ft ffcg -fc fred
time python find_shortest_paths.py -ft tfcg -fc fred