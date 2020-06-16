#!/bin/bash
#SBATCH -J create_leave1out_ffcg
#SBATCH -p general
#SBATCH -o create_leave1out_ffcg_%j.txt
#SBATCH -e create_leave1out_ffcg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=24G
#SBATCH --time=96:00:00
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_leave1out.py -fc fred -ft ffcg -cpu 48