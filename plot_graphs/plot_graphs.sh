#!/bin/bash
#SBATCH -J plot_graphs
#SBATCH -p general
#SBATCH -o plot_graphs_%j.txt
#SBATCH -e plot_graphs_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=58G
#SBATCH --time=00:15:00
errcho(){ >&2 echo $@; }
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
################################################################
errcho order_paths
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
time python order_paths.py -fcg fred -ft tfcg
time python order_paths.py -fcg fred -ft ffcg
time python order_paths.py -fcg fred
################################################################
errcho plots
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/plot_graphs/
time python plot_sp.py -fcg fred -ft tfcg -pt roc
time python plot_sp.py -fcg fred -ft tfcg -pt dist
time python plot_sp.py -fcg fred -ft ffcg -pt roc
time python plot_sp.py -fcg fred -ft ffcg -pt dist
time python plot_sp.py -fcg fred -pt roc
time python plot_sp.py -fcg fred -pt dist
# time python plot_adj_pairs.py -fcg fred -kg dbpedia
# time python plot_adj_pairs.py -fcg fred1 -kg dbpedia
# time python plot_adj_pairs.py -fcg fred2 -kg dbpedia
# time python plot_adj_pairs.py -fcg fred3 -kg dbpedia 
# time python plot_adj_pairs.py -fcg backbone_df -kg dbpedia 
# time python plot_adj_pairs.py -fcg largest_ccf -kg dbpedia 
# time python plot_adj_pairs.py -fcg old_fred -kg dbpedia 