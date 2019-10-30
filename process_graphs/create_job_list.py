import os
import codecs
import json
import numpy as np 
import pandas as pd 
import argparse

def create_job_list(graph_path,kg_label,fcg_class,total_t,n):
	kg_datapath=os.path.join(graph_path,"kg",kg_label,"data")
	jobs_dir="intersect_all_entityPairs_{}_{}_jobs".format(kg_label,fcg_class)
	os.makedirs(os.path.join(kg_datapath,jobs_dir),exist_ok=True)
	with codecs.open(os.path.join(kg_datapath,"intersect_all_entityPairs_{}_{}_{}_IDs_klformat.txt".format(kg_label,fcg_class,kg_label)),"r","utf-8") as f:
		file=f.readlines()
	t=int(total_t/n)
	nlines=int(len(file)/n)+1
	for i in range(n):
		file_i=file[nlines*i:min(nlines*(i+1),len(file))]
		inout=os.path.join(jobs_dir,"{}_intersect_all_entityPairs_{}_{}_job".format(str(i+1),kg_label,fcg_class))
		with codecs.open(os.path.join(kg_datapath,inout+".txt"),"w","utf-8") as f:
			for line in file_i:
				f.write(line)
		with codecs.open(os.path.join(kg_datapath,jobs_dir,str(i+1)+"_klinker.sh"),"w","utf-8") as f:
			f.write('''
#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=180gb,walltime={}:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N {}_klinker
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd {}/
klinker linkpred {} {} {}.txt {}.json -u -n 12 -w logdegree
				'''.format(t,str(i+1),kg_datapath,kg_label+"_nodes.txt",kg_label+"_edgelistID.npy",inout,inout))

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create klinker job list')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	parser.add_argument('-kg','--kgtype', metavar='knowledgegraph type',type=str,choices=['dbpedia','wikidata'],help='Choose KnowledgeGraph Type')
	parser.add_argument('-fcg','--fcgclass', metavar='factcheckgraph class',type=str,choices=['fred','co-occur','kg','backbone-dbpedia-fred','backbone-dbpedia-co-occur'],help='Choose FactCheckGraph Class')
	parser.add_argument('-t','--totalt', metavar='total time',type=int,help='Total Time in hours')
	parser.add_argument('-n','--splits', metavar='split number',type=int,help='Number of Splits')
	args=parser.parse_args()
	create_job_list(args.graphpath,args.kgtype,args.fcgclass,args.totalt,args.splits)