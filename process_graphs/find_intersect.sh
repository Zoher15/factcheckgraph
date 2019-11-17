#!/bin/bash

#SBATCH -J find_intersect
#SBATCH -p general
#SBATCH -o find_intersect_%j.txt
#SBATCH -e find_intersect_p_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/process_graphs/
# time python find_intersect.py -fcg fred -kg dbpedia
time python find_intersect.py -fcg fred1 -kg dbpedia
time python find_intersect.py -fcg fred2 -kg dbpedia
time python find_intersect.py -fcg fred3 -kg dbpedia
# time python find_intersect.py -fcg co-occur -kg dbpedia
# time python find_intersect.py -fcg backbone_df -kg dbpedia
# time python find_intersect.py -fcg backbone_dc -kg dbpedia
# time python find_intersect.py -fcg largest_ccf -kg dbpedia
# time python find_intersect.py -fcg largest_ccc -kg dbpedia
# time python find_intersect.py -fcg old_fred -kg dbpedia