#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=16gb,walltime=5:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N find_shortest_paths2
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
time python find_shortest_paths.py -ft ffcg -mp roberta-base-nli-stsb-mean-tokens
time python find_shortest_paths.py -ft tfcg -mp roberta-base-nli-stsb-mean-tokens