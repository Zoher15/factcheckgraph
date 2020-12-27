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
#SBATCH --time=01:00:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/plot_graphs/
time python plot_sp.py -fcg fred -ft tfcg -pt roc -gt undirected
time python plot_sp.py -fcg fred -ft tfcg -pt dist -gt undirected
# time python plot_adj_pairs.py -fcg fred -kg dbpedia
# time python plot_adj_pairs.py -fcg fred1 -kg dbpedia
# time python plot_adj_pairs.py -fcg fred2 -kg dbpedia
# time python plot_adj_pairs.py -fcg fred3 -kg dbpedia 
# time python plot_adj_pairs.py -fcg backbone_df -kg dbpedia 
# time python plot_adj_pairs.py -fcg largest_ccf -kg dbpedia 
# time python plot_adj_pairs.py -fcg old_fred -kg dbpedia 