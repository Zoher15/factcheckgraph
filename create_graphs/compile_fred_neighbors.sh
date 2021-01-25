#!/bin/bash
#SBATCH -J compile_fred_neighbors
#SBATCH -p general
#SBATCH -o compile_fred_neighbors_%j.txt
#SBATCH -e compile_fred_neighbors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
errcho(){ >&2 echo $@; }
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph/create_graphs/
################################################################
errcho fetching
time python fetch_fred.py -ft tfcg -cpu 48 -p
time python fetch_fred.py -ft ffcg -cpu 48 -p 
################################################################
errcho compiling
time python compile_fred.py -ft ufcg -cpu 48 -n 10
time python compile_fred.py -ft ufcg -cpu 48 -n 20
time python compile_fred.py -ft ufcg -cpu 48 -n 30
time python compile_fred.py -ft ufcg -cpu 48 -n 40
time python compile_fred.py -ft ufcg -cpu 48 -n 50
time python compile_fred.py -ft ufcg -cpu 48 -n 60
time python compile_fred.py -ft ufcg -cpu 48 -n 70
time python compile_fred.py -ft ufcg -cpu 48 -n 80
time python compile_fred.py -ft ufcg -cpu 48 -n 90
time python compile_fred.py -ft ufcg -cpu 48 -n 100

