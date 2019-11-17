#!/bin/bash

#SBATCH -J find_adj_pairs
#SBATCH -p general
#SBATCH -o find_adj_pairs_%j.txt
#SBATCH -e find_adj_pairs_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
# time python find_adj_pairs.py -fcg fred -kg dbpedia
time python find_adj_pairs.py -fcg fred1 -kg dbpedia
time python find_adj_pairs.py -fcg fred2 -kg dbpedia
time python find_adj_pairs.py -fcg fred3 -kg dbpedia
# time python find_adj_pairs.py -fcg backbone_df -kg dbpedia 
# time python find_adj_pairs.py -fcg largest_ccf -kg dbpedia 
# time python find_adj_pairs.py -fcg old_fred -kg dbpedia 
