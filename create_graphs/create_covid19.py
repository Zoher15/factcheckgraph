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

def create_co_occur(rdf_path,graph_path):
	claim_type="covid19"
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
	claim_IDs=claims['claimID'].tolist()
	claim_entities={}
	claim_edges={}
	entity_regex=re.compile(r'http:\/\/dbpedia\.org')
	fcg_co=nx.MultiGraph()
	init=0
	init2=0
	for claim_ID in claim_IDs:
		claim_entities_set=set([])
		claim_g=rdflib.Graph()
		claim_nxg=nx.MultiGraph()
		filename="claim{}.rdf".format(str(claim_ID))
		try:
			claim_g.parse(os.path.join(rdf_path,"{}_claims".format(claim_type),filename),format='application/rdf+xml')
		except:
			pass
		for triple in claim_g:
			subject,predicate,obj=list(map(str,triple))
			subject=subject.replace("Middle_East_respiratory_syndrome_coronavirus","Severe_acute_respiratory_syndrome_coronavirus_2")
			obj=obj.replace("Middle_East_respiratory_syndrome_coronavirus","Severe_acute_respiratory_syndrome_coronavirus_2")
			try:
				if entity_regex.search(subject):
					claim_entities_set.add(subject)
				if entity_regex.search(obj):
					claim_entities_set.add(obj)
			except KeyError:
				pass
		claim_entities[claim_ID]=list(claim_entities_set)
		init+=len(claim_entities_set)
		claim_edges[claim_ID]=list(combinations(claim_entities[claim_ID],2))
		init2+=len(claim_edges[claim_ID])
		for edge in claim_edges[claim_ID]:
			claim_nxg.add_edge(edge[0],edge[1],claim_ID=claim_ID)
		filename=os.path.join(rdf_path,"{}_claims".format(claim_type),"claim{}_co".format(str(claim_ID)))
		nx.write_edgelist(claim_nxg,filename+".edgelist")
		before=len(fcg_co.edges)
		fcg_co.add_edges_from(claim_nxg.edges.data())
	print(float(init)/len(claim_IDs))
	print(init2)
	fcg_path=os.path.join(graph_path,"co-occur")
	os.makedirs(fcg_path, exist_ok=True)
	nx.write_edgelist(fcg_co,os.path.join(fcg_path,"{}.edgelist".format(claim_type)))
	nx.write_graphml(fcg_co,os.path.join(fcg_path,"{}.graphml".format(claim_type)),prettyprint=True)
	os.makedirs(os.path.join(fcg_path,"data"),exist_ok=True)
	write_path=os.path.join(fcg_path,"data")
	nodes=list(fcg_co.nodes)
	edges=list(fcg_co.edges)
	#Save Nodes
	with codecs.open(write_path+"_nodes.txt","w","utf-8") as f:
		for node in nodes:
			f.write(str(node)+"\n")
	#Save Entities
	entity_regex=re.compile(r'http:\/\/dbpedia\.org')
	entities=np.asarray([node for node in nodes if entity_regex.search(node)])
	with codecs.open(write_path+"_entities.txt","w","utf-8") as f:
		for entity in entities:
			f.write(str(entity)+"\n")
	#Save node2ID dictionary
	node2ID={node:i for i,node in enumerate(nodes)}
	with codecs.open(write_path+"_node2ID.json","w","utf-8") as f:
		f.write(json.dumps(node2ID,ensure_ascii=False))
	#Save Edgelist ID
	edgelistID=np.asarray([[node2ID[edge[0]],node2ID[edge[1]],1] for edge in edges])
	np.save(write_path+"_edgelistID.npy",edgelistID)

def assign_weight(p,rdf_path,graph_path,emb_path):
	claim_type="covid19"
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
		if not fcg_co.has_edge(u,v)
			if len(fcg_co.get_edge_data(u,v))==1:
				fcg_co2.add_edge(u,v)
				fcg_co2.edges[u,v].update(data)
			else:
				datalist=fcg_co.get_edge_data(u,v)
				fcg_co2.add_edge(u,v)
				#finding the data dict among all the multiedges between u and v with the max similarity (or min weight)
				data=max(fcg_co.get_edge_data(u,v).items(), key=lambda x:x[1]['simil'])[1]
				fcg_co2.edges[u,v].update(data)

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create co-cccur graph')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/covid19_rdf_files/")
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/graphs/covid19/")
	args=parser.parse_args()
	create_co_occur(args.rdfpath,args.graphpath)