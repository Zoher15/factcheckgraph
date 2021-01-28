#!/bin/bash
#SBATCH -J find_shortest_paths
#SBATCH -p general
#SBATCH -o find_shortest_paths_%j.txt
#SBATCH -e find_shortest_paths_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zoher.kachwala@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=58G
#SBATCH --time=24:00:00
errcho(){ >&2 echo $@; }
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
for var in 1 10 20 30 40 50 60 70 80 90 100 110 150 180 190 200 230 250 290 300 310 330 350 400
do
	errcho $var
	cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
	################################################################
	errcho find_shortest_paths
	time python find_shortest_paths.py -st tfcg -ft tfcg -mp roberta-base-nli-stsb-mean-tokens -cpu 48 -n $var
	time python find_shortest_paths.py -st tfcg -ft ffcg -mp roberta-base-nli-stsb-mean-tokens -cpu 48 -n $var
	time python order_paths.py -fcg fred -ft tfcg
	################################################################
	errcho plot graphs 
	cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/plot_graphs/
	time python plot_sp.py -fcg fred -ft tfcg -pt roc -pr n$var
	time python plot_sp.py -fcg fred -ft tfcg -pt dist -pr n$var
	################################################################
	cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/analyze_graphs/
	################################################################
	errcho find_shortest_paths
	time python find_shortest_paths.py -st ffcg -ft ffcg -mp roberta-base-nli-stsb-mean-tokens -cpu 48 -n $var
	time python find_shortest_paths.py -st ffcg -ft tfcg -mp roberta-base-nli-stsb-mean-tokens -cpu 48 -n $var
	time python order_paths.py -fcg fred -ft ffcg
	time python order_paths.py -fcg fred
	################################################################
	errcho plot graphs 
	cd /geode2/home/u110/zkachwal/BigRed3/factcheckgraph/plot_graphs/
	time python plot_sp.py -fcg fred -ft ffcg -pt roc -pr n$var
	time python plot_sp.py -fcg fred -ft ffcg -pt dist -pr n$var
	time python plot_sp.py -fcg fred -pt roc -pr n$var
	time python plot_sp.py -fcg fred -pt dist -pr n$var
done