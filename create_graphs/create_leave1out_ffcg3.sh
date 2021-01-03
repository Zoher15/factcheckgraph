#!/bin/bash
#SBATCH -J create_leave1out_ffcg3
#SBATCH -p general
#SBATCH -o create_leave1out_ffcg3_%j.txt
#SBATCH -e create_leave1out_ffcg3_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=58G
#SBATCH --cpus-per-task=48
#SBATCH --time=30:00:00
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/create_graphs/
time python create_leave1out.py -fc fred -ft ffcg -cpu 48 -jobs 4 -jn 3