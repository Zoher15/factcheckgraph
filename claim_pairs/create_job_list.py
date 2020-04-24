import os
import codecs
import json
import numpy as np 
import pandas as pd 
import argparse
import time

def create_job_list(graph_path,pairs_path,kg_label,total_t,n):
	kg_datapath=os.path.join(graph_path,"kg",kg_label,"data")
	jobs_path=os.path.join(pairs_path,"jobs")
	os.makedirs(jobs_path,exist_ok=True)
	with codecs.open(os.path.join(pairs_path,"data",'intersect_claims_entityPairs_{}_{}_{}_IDs_klformat.txt'.format(kg_label,'claims',kg_label)),"r","utf-8") as f:
		file=f.readlines()
	t=(float(total_t)/n)*3600
	nlines=int(len(file)/n)+1
	for i in range(n):
		file_i=file[nlines*i:min(nlines*(i+1),len(file))]
		inout=os.path.join(jobs_path,"{}_claim_pairs_{}_job".format(str(i+1),kg_label))
		with codecs.open(os.path.join(jobs_path,inout+".txt"),"w","utf-8") as f:
			for line in file_i:
				f.write(line)
		with codecs.open(os.path.join(jobs_path,str(i+1)+"_klinker.sh"),"w","utf-8") as f:
			f.write('''
#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=200gb,walltime={}
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N {}_claim_pairs_klinker
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd {}/
klinker linkpred {} {} {}.txt {}.json -u -n 12 -w logdegree
				'''.format(time.strftime('%H:%M:%S',time.gmtime(t)),str(i+1),kg_datapath,kg_label+"_nodes.txt",kg_label+"_edgelistID.npy",inout,inout))

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create klinker job list')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-pp','--pairspath', metavar='pairs path',type=str,help='Directory for the claim pairs',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/claim_pairs/')
	parser.add_argument('-kg','--kgtype', metavar='knowledgegraph type',type=str,choices=['dbpedia','wikidata'],help='Choose KnowledgeGraph Type')
	parser.add_argument('-t','--totalt', metavar='total time',type=int,help='Total Time in hours')
	parser.add_argument('-n','--splits', metavar='split number',type=int,help='Number of Splits')
	args=parser.parse_args()
	create_job_list(args.graphpath,args.pairspath,args.kgtype,args.totalt,args.splits)