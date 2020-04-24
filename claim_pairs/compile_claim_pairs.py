import os
import codecs
import json
import numpy as np 
import pandas as pd 
import argparse
import time

def compile_claim_pairs(pairs_path,kg_label,n):
	claim_pairs_data=pd.read_json(os.path.join(pairs_path,"{}_claim_pairs_{}_job.json".format(str(1),kg_label)))
	print(len(claim_pairs_data))
	for i in range(2,n+1):
		claim_pairs_data=claim_pairs_data.append(pd.read_json(os.path.join(pairs_path,"{}_claim_pairs_{}_job.json".format(str(i),kg_label))),ignore_index=True)
		print(len(claim_pairs_data))
	import pdb
	pdb.set_trace()

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create klinker job list')
	# parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-pp','--pairspath', metavar='pairs path',type=str,help='Directory for the claim pairs',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/claim_pairs/jobs/')
	parser.add_argument('-kg','--kgtype', metavar='knowledgegraph type',type=str,choices=['dbpedia','wikidata'],help='Choose KnowledgeGraph Type')
	parser.add_argument('-n','--splits', metavar='split number',type=int,help='Number of Splits')
	args=parser.parse_args()
	compile_claim_pairs(args.pairspath,args.kgtype,args.splits)