#!/bin/bash
#SBATCH -J find_baseline
#SBATCH -p general
#SBATCH -o find_baseline_%j.txt
#SBATCH -e find_baseline_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=58G
#SBATCH --time=1:00:00

source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
time python embed.py -ft ffcg -fc co_occur -gt undirected
time python embed.py -ft tfcg -fc co_occur -gt undirected
time python embed.py -ft ffcg -fc fred -gt undirected
time python embed.py -ft tfcg -fc fred -gt undirected
time python embed.py -ft ffcg -fc co_occur -mp roberta-base-nli-stsb-mean-tokens -gt undirected
time python embed.py -ft tfcg -fc co_occur -mp roberta-base-nli-stsb-mean-tokens -gt undirected
time python embed.py -ft ffcg -fc fred -mp roberta-base-nli-stsb-mean-tokens -gt undirected
time python embed.py -ft tfcg -fc fred -mp roberta-base-nli-stsb-mean-tokens -gt undirected
time python find_baseline.py -bt knn -cpu 48
time python find_baseline.py -bt all