#!/bin/bash
#SBATCH -J create_leave1out
#SBATCH -p general
#SBATCH -o create_leave1out_%j.txt
#SBATCH -e create_leave1out_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=58G
#SBATCH --cpus-per-task=48
#SBATCH --time=120:00:00
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/create_graphs/
# time python create_leave1out.py -fc fred -ft tfcg -cpu 48
# time python create_leave1out.py -fc fred -ft ffcg -cpu 48
# time python create_leave1out.py -fc co_occur -ft tfcg_co -cpu 48
time python create_leave1out.py -fc co_occur -ft ffcg_co -cpu 48