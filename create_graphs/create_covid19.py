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
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing as mp
from collections import ChainMap 
from itertools import chain
from sentence_transformers import SentenceTransformer
from create_fred import *
from create_co_occur import *

def create_weighted(p,rdf_path,graph_path,emb_path,claim_type):
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
	claim_IDs=claims['claimID'].tolist()
	claims_emb=pd.read_csv(os.path.join(emb_path,"claims_embeddings.tsv"),delimiter="\t",header=None).values
	simil_p=cosine_similarity(p,claims_emb)
	fcg_path=os.path.join(graph_path,"co-occur")
	fcg_co=nx.read_edgelist(os.path.join(fcg_path,"{}.edgelist".format(claim_type)),comments="@",create_using=nx.MultiGraph)
	#assigning weight by summing the log of adjacent nodes, and dividing by the similiarity of the claim with the target predicate
	for u,v,k,claimID in fcg_co.edges.data('claim_ID',keys=True):
		claimIX=claims[claims['claimID']==claimID].index[0]
		uw=np.log10(fcg_co.degree(u))
		vw=np.log10(fcg_co.degree(v))
		weight=(uw+vw)/simil_p[claimIX]
		fcg_co.edges[u,v,k]['simil']=simil_p[claimIX]
		fcg_co.edges[u,v,k]['weight']=weight
	fcg_co2=nx.Graph()
	for u,v,k,data in fcg_co.edges.data(keys=True):
		if not fcg_co.has_edge(u,v):
			if len(fcg_co.get_edge_data(u,v))==1:
				fcg_co2.add_edge(u,v)
				fcg_co2.edges[u,v].update(data)
			else:
				datalist=fcg_co.get_edge_data(u,v)
				fcg_co2.add_edge(u,v)
				#finding the data dict among all the multiedges between u and v with the max similarity (or min weight)
				data=max(fcg_co.get_edge_data(u,v).items(), key=lambda x:x[1]['simil'])[1]
				fcg_co2.edges[u,v].update(data)
	return fcg_co2

def create_fred(rdf_path,claim_type,passive,cpu):
	claims_path=os.path.join(rdf_path,"{}_claims".format(claim_type))
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
	claim_IDs=claims['claimID'].tolist()
	init=0
	end=len(claims)
	if not passive:
		errorclaimid,clean_claims=fredParse(claims_path,claims,init,end)
	else:
		n=int(len(claim_IDs)/cpu)+1
		pool=mp.Pool(processes=cpu)							
		results=[pool.apply_async(passiveFredParse, args=(index,claims_path,claim_IDs,index*n,(index+1)*n)) for index in range(cpu)]
		output=sorted([p.get() for p in results],key=lambda x:x[0])
		errorclaimid=list(chain(*map(lambda x:x[1],output)))
		clean_claims=dict(ChainMap(*map(lambda x:x[2],output)))
	np.save(os.path.join(rdf_path,"{}_error_claimID.npy".format(claim_type)),errorclaimid)
	with codecs.open(os.path.join(rdf_path,"{}claims_clean.json".format(claim_type)),"w","utf-8") as f:
		f.write(json.dumps(clean_claims,indent=4,ensure_ascii=False))

def embed_claims(rdf_path,model_path,embed_path,claim_type):
	model = SentenceTransformer(model_path)
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
	claims_list=list(claims['claim_text'])
	claims_embeddings=model.encode(claims_list)
	with open(os.path.join(embed_path,'{}_claims_embeddings.tsv'.format(claim_type)),'w',newline='') as f:
		for vector in claims_embeddings:
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)
	with open(os.path.join(embed_path,'{}_claims_embeddings_labels.tsv'.format(claim_type)),'w',newline='') as f:
		vector=['claim_text','rating','claimID']
		tsv_output=csv.writer(f,delimiter='\t')
		tsv_output.writerow(vector)
		for i in range(len(claims)):
			vector=list((claims.iloc[i]['claim_text'],claims.iloc[i]['factcheck_rating'],claims.iloc[i]['claimID']))
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)

def create_fred_target():
	pass

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create co-cccur graph')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/covid19_rdf_files/")
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/graphs/covid19/")
	parser.add_argument('-mp','--modelpath', metavar='model path',type=str,help='Model directory to load the model',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/models/covid19-relatedness-model/claims-roberta-base-nli-mean-tokens-2020-03-29_19-36-45")
	parser.add_argument('-ep','--embedpath', metavar='embed path',type=str,help='Model directory to save and load embeddings',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/embeddings")
	parser.add_argument('-p','--passive',action='store_true',help='Passive or not',default=False)
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,help='Custom FactCheckGraph',default="covid19")
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	args=parser.parse_args()
	import pdb
	pdb.set_trace()
	# create_fred(args.rdfpath,args.fcgtype,args.passive,args.cpu)
	# create_co_occur(args.rdfpath,args.graphpath,args.fcgtype)
	# embed_claims(args.rdfpath,args.modelpath,args.embedpath,args.fcgtype)