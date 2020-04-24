import os
import pandas as pd
import networkx as nx
import re
import numpy as np
import argparse
import rdflib
import codecs
import json
from itertools import combinations

def save_claim_pairs(rdf_path,graph_path,pairs_path,kg_label):
	fcg_labels=['tfcg_co','ffcg_co']
	pairs_set=set([])
	kg_path=os.path.join(graph_path,"kg",kg_label,"data",kg_label)
	#Fetching kg entities
	kg_entities=set(list(np.loadtxt(kg_path+"_entities.txt",dtype='str',encoding='utf-8')))
	#Reading the kg node2ID dict
	with codecs.open(kg_path+"_node2ID.json","r","utf-8") as f:
		node2ID=json.loads(f.read())
	#Unioning edges in both graphs
	for fcg_label in fcg_labels:
		fcg_path=os.path.join(graph_path,"co-occur",fcg_label)
		fcg_co=nx.read_edgelist(os.path.join(fcg_path,"{}.edgelist".format(fcg_label)),comments="@")
		pairs_set=pairs_set.union(set(map(lambda x:str(tuple(sorted(x))),list(fcg_co.edges()))))
	pairs=sorted(list(pairs_set))
	absent_pairs=[]
	pairs_klformat=[]
	pairs_path_data=os.path.join(pairs_path,"data")
	os.makedirs(pairs_path_data,exist_ok=True)
	with codecs.open(os.path.join(pairs_path,"claim_pairs.txt"),"w","utf-8") as f1:
		with codecs.open(os.path.join(pairs_path_data,'intersect_claims_entityPairs_{}_{}_{}_IDs_klformat.txt'.format(kg_label,'claims',kg_label)),"w","utf-8") as f2:
			for pair in pairs:
				epair=eval(pair)
				if set(epair).issubset(kg_entities):
					f1.write("{}\n".format(pair))
					f2.write("{} {} {} {} {} {} {}\n".format(str(np.nan),str(int(node2ID[epair[0]])),str(np.nan),str(np.nan),str(int(node2ID[epair[1]])),str(np.nan),str(np.nan)))
				else:
					absent_pairs.append(pair)
	with codecs.open(os.path.join(pairs_path,"absent_pairs.txt"),"w","utf-8") as f:
		for pair in absent_pairs:
			f.write("{}\n".format(pair))	

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create co-cccur graph')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/rdf_files/')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-pp','--pairspath', metavar='pairs path',type=str,help='True False or Union FactCheckGraph',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/claim_pairs/')
	parser.add_argument('-kg','--kg', metavar='knowledgegraph type',type=str,choices=['dbpedia','wikidata'],help='DBPedia or Wikidata Graph')
	args=parser.parse_args()
	save_claim_pairs(args.rdfpath,args.graphpath,args.pairspath,args.kg)