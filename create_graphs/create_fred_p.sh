#!/bin/bash

#SBATCH -J create_fred_p
#SBATCH -p general
#SBATCH -o create_fred_p_%j.txt
#SBATCH -e create_fred_p_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:00:00
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
# time python create_fred.py -ft tfcg -p -cpu 10
# time python create_fred.py -ft ffcg -p -cpu 10
# time python create_fred.py -ft ufcg
# time python create_backbone.py -fcg fred -ft tfcg -kg dbpedia
# time python create_backbone.py -fcg fred -ft ffcg -kg dbpedia
# time python create_backbone.py -fcg fred -ft ufcg -kg dbpedia
# time python create_largest_cc.py -fcg fred -ft tfcg 
# time python create_largest_cc.py -fcg fred -ft ffcg
# time python create_largest_cc.py -fcg fred -ft ufcg 
%run create_fred.py -ft tfcg -cpu 20 -cf 1
%run create_fred.py -ft ffcg -cpu 20 -cf 1
%run create_fred.py -ft tfcg -cpu 20 -cf 2
%run create_fred.py -ft ffcg -cpu 20 -cf 2
%run create_fred.py -ft tfcg -cf 3
%run create_fred.py -ft ffcg -cf 3
