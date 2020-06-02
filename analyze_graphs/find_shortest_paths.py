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
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import multiprocessing as mp
from collections import ChainMap 
from itertools import chain
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
from statistics import mean
import sys
sys.path.insert(1, '/geode2/home/u110/zkachwal/BigRed3/factcheckgraph/create_graphs')
from create_fred import *
from create_co_occur import *
import csv

def chunkstring(string,length):
	chunks=int(len(string)/length)+1
	offset=0
	for i in range(1,chunks):
		if i*length<len(string):
			if string[i*length]==' ':
				loc=i*length
			else:
				left=string[:i*length][::-1].find(" ")
				right=string[i*length:].find(" ")
				if left<right:
					loc=i*length-left-1
				else:
					loc=i*length+right
			string=string[:loc]+"\n"+string[loc+1:]
	return string.splitlines()

def embed_claims(rdf_path,model_path,embed_path,claim_type):
	os.makedirs(embed_path, exist_ok=True)
	model = SentenceTransformer(model_path)
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
	claims_list=list(claims['claim_text'])
	claims_embeddings=model.encode(claims_list)
	with open(os.path.join(embed_path,'{}_claims_embeddings.tsv'.format(claim_type)),'w',newline='') as f:
		for vector in claims_embeddings:
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)
	with open(os.path.join(embed_path,'{}_claims_embeddings_labels.tsv'.format(claim_type)),'w',newline='') as f:
		vector=['claim_text','claimID','rating']
		tsv_output=csv.writer(f,delimiter='\t')
		tsv_output.writerow(vector)
		for i in range(len(claims)):
			vector=list((claims.iloc[i]['claim_text'],claims.iloc[i]['claimID'],claim_type))
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)

def create_weighted(p,rdf_path,model_path,graph_path,embed_path,claim_type,fcg_type):
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
	claim_IDs=claims['claimID'].tolist()
	try:
		claims_embed=pd.read_csv(os.path.join(embed_path,"{}_claims_embeddings.tsv".format(claim_type)),delimiter="\t",header=None).values
	except FileNotFoundError:
		embed_claims(rdf_path,model_path,embed_path,claim_type)
		claims_embed=pd.read_csv(os.path.join(embed_path,"{}_claims_embeddings.tsv".format(claim_type)),delimiter="\t",header=None).values
	simil_p=cosine_similarity(p,claims_embed)[0]
	#angular distance
	simil_p=1-np.arccos(simil_p)/np.pi
	if fcg_type not in set(['tfcg_co','ffcg_co']):
		fcg_class=fcg_type.split('-')[0]
		fcg_path=os.path.join(graph_path,"co-occur",fcg_class,"graphs3")
	else:
		fcg_path=os.path.join(graph_path,"co-occur",fcg_type)
	fcg_co=nx.read_edgelist(os.path.join(fcg_path,"{}.edgelist".format(fcg_type)),comments="@",create_using=nx.MultiGraph)
	#assigning weight by summing the log of adjacent nodes, and dividing by the similiarity of the claim with the target predicate
	for u,v,k,claimID in fcg_co.edges.data('claim_ID',keys=True):
		claimIX=claims[claims['claimID']==claimID].index[0]
		uw=np.log10(fcg_co.degree(u))
		vw=np.log10(fcg_co.degree(v))
		if simil_p[claimIX]>0:
			dist=float(1)-simil_p[claimIX]
			weight=(uw+vw)*dist*0.5
		else:
			dist=np.inf
			weight=np.inf
		fcg_co.edges[u,v,k]['dist']=dist
		fcg_co.edges[u,v,k]['weight']=weight
	fcg_co2=nx.Graph()
	for u,v,k,data in fcg_co.edges.data(keys=True):
		if not fcg_co2.has_edge(u,v):
			if len(fcg_co.get_edge_data(u,v))==1:
				fcg_co2.add_edge(u,v)
				fcg_co2.edges[u,v].update(data)
			else:
				datalist=fcg_co.get_edge_data(u,v)
				fcg_co2.add_edge(u,v)
				#finding the data dict among all the multiedges between u and v with the min distance
				data=min(fcg_co.get_edge_data(u,v).items(), key=lambda x:x[1]['dist'])[1]
				fcg_co2.edges[u,v].update(data)
	return fcg_co2

def aggregate_edge_data(evalues,mode):
	#mode can be dist or weight
	edgepair_weights=[]
	for edgepair,e2values in evalues.items():
		edgepair_weights.append(e2values[mode])
	return sum(edgepair_weights)

def aggregate_weights(claim_D,mode,mode2):
	#mode can be max, min, sum, mean
	#mode2 can be w or s
	edge_weights=[]
	for edge,evalues in claim_D.items():
		if type(evalues)!=list:
			u,v,w,d=eval(edge.replace("inf","np.inf"))
			edge_weights.append(eval(mode2))
	return eval("{}(edge_weights)".format(mode))

def create_ordered_paths(write_path,mode):
	#mode can be weight/dist
	rw_path=write_path+"_{}".format(mode)
	with codecs.open(rw_path+".json","r","utf-8") as f: 
		paths_w=json.loads(f.read())
	for mode2 in ['sum','mean','max','min']:
		ordered_paths=OrderedDict(sorted(paths_w.items(), key=lambda t: aggregate_weights(t[1],mode2,mode)))
		with codecs.open(rw_path+"_{}.json".format(mode2),"w","utf-8") as f:
			f.write(json.dumps(ordered_paths,indent=5,ensure_ascii=False))

#Function to find node pairs in the source graph if they exist as edges in the target graph
def source_target(rdf_path,graph_path,embed_path,source_fcg_type,target_fcg_type,mode):
	suffix={"co-occur":"_co","fred":""}
	if target_fcg_type in source_fcg_type:
		fcg_class,target_claimID=source_fcg_type.split('-')
		# fcg_class=source_fcg_type.split('-')[0]
		claim_types={'tfcg_co':'true','ffcg_co':'false','tfcg':'true','ffcg':'false'}
		claim_type=claim_types[fcg_class]+"_claims"
		source_fcg_path=os.path.join(graph_path,mode,fcg_class,"graphs3")
		target_fcg_path=os.path.join(rdf_path,claim_type)
		target_fcg_type="claim"+target_claimID+suffix[mode]
	else:
		source_fcg_path=os.path.join(graph_path,mode,source_fcg_type)
		target_fcg_path=os.path.join(graph_path,mode,target_fcg_type)
	#loading source and target fcg
	source_fcg=nx.read_edgelist(os.path.join(source_fcg_path,"{}.edgelist".format(source_fcg_type)),comments="@",create_using=nx.MultiGraph)
	target_fcg=nx.read_edgelist(os.path.join(target_fcg_path,"{}.edgelist".format(target_fcg_type)),comments="@",create_using=nx.MultiGraph)
	edges_of_interest={}
	'''
	If the graph is co-occurrence, then the target edges of interest are simply all the edges in the graph.
	But if it is Fred, then the target edges of interest are more complicated.
	It has to be the paths between entities (which do not contain other entities)
	'''
	edges=[]
	if mode=='co-occur':
		edges=target_fcg.edges.data(keys=True)
	else:
		nodes=list(target_fcg.nodes)
		entity_regex=re.compile(r'db:.*')
		entities=list(filter(lambda x:entity_regex.search(x),nodes))
		entitypairs=list(combinations(entities,2))
		#limiting edges to entity pairs that have a path between them in the target graph
		edges=list(filter(lambda x:nx.has_path(target_fcg,x[0],x[1]),entitypairs))
	for edge in edges:
		u,v,k,data=edge
		if source_fcg.has_node(u) and source_fcg.has_node(v) and nx.has_path(source_fcg,u,v):#and not source_fcg.has_edge(u,v) and 
			try:
				edges_of_interest[data['claim_ID']].add((u,v))
			except KeyError:
				edges_of_interest[data['claim_ID']]=set([(u,v)])
	return edges_of_interest

'''
Function does the following
1. Finds node pairs of interest from graph built through target claims by using the function source_target
2. Creates a weighted source graph for each target claim using the function create_weighted
3. Finds shortest path for each edge of interest in the target graph
'''
def shortest_paths(rdf_path,model_path,graph_path,embed_path,source_fcg_type,target_fcg_type,mode):
	fcg_types={'co-occur':{'tfcg':'tfcg_co','ffcg':'ffcg_co'},'fred':{'tfcg':'tfcg','ffcg':'ffcg'}}
	source_fcg_type=fcg_types[mode][source_fcg_type]
	target_fcg_type=fcg_types[mode][target_fcg_type]
	claim_types={'tfcg_co':'true','ffcg_co':'false','tfcg':'true','ffcg':'false'}
	source_claim_type=claim_types[source_fcg_type]
	target_claim_type=claim_types[target_fcg_type]
	#path to store the shortest paths
	write_path=os.path.join(graph_path,mode,"paths",source_fcg_type+"_"+target_claim_type+"_"+model_path.split("/")[-1])
	os.makedirs(write_path,exist_ok=True)
	write_path=os.path.join(write_path,"paths")
	#path to store the weghted graphs
	writegraph_path=os.path.join(graph_path,"weighted","graphs_"+source_fcg_type+"_"+target_claim_type+"_"+model_path.split("/")[-1])
	os.makedirs(writegraph_path,exist_ok=True)
	#path for the graph that is being used to fact-check
	source_fcg_path=os.path.join(graph_path,mode,source_fcg_type)
	#loading existing path files
	try:
		with codecs.open(write_path+"_w.json","r","utf-8") as f:
			paths_of_interest_w=json.loads(f.read())
	except FileNotFoundError:
		paths_of_interest_w={}
	try:
		with codecs.open(write_path+"_d.json","r","utf-8") as f:
			paths_of_interest_d=json.loads(f.read())
	except FileNotFoundError:
		paths_of_interest_d={}
	#loading claims
	source_claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(source_claim_type)))
	target_claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(target_claim_type)))
	#loading target claim embeddings
	try:
		target_claims_embed=pd.read_csv(os.path.join(embed_path,"{}_claims_embeddings.tsv".format(target_claim_type)),delimiter="\t",header=None).values
	except FileNotFoundError:
		embed_claims(rdf_path,model_path,embed_path,target_claim_type)
		target_claims_embed=pd.read_csv(os.path.join(embed_path,"{}_claims_embeddings.tsv".format(target_claim_type)),delimiter="\t",header=None).values
	#limiting claimIDs to edges of interest if both graphs are different
	if source_fcg_type==target_fcg_type:
		claimIDs=source_claims['claimID'].tolist()
	else:
		edges_of_interest=source_target(rdf_path,graph_path,embed_path,source_fcg_type,target_fcg_type)
		claimIDs=list(edges_of_interest.keys())
	#iterating throuhg the claimIDs of interest
	for claimID in claimIDs:
		if source_fcg_type==target_fcg_type:
			#checking if the source graph (without the given claim), has edges of interest. If not, we skip the claim .i.e continue in the for loop
			edges_of_interest=source_target(rdf_path,graph_path,embed_path,source_fcg_type+'-'+str(claimID),target_fcg_type)
			if claimID not in set(list(edges_of_interest.keys())):
				continue
		#getting index of claim
		claimIX=target_claims[target_claims['claimID']==claimID].index[0]
		p=np.array([target_claims_embed[claimIX]])
		#source_fcg is not a multigraph
		#if source and target are the same graph, then the graph without the given claim is fetched
		if source_fcg_type==target_fcg_type:
			source_fcg=create_weighted(p,rdf_path,model_path,graph_path,embed_path,source_claim_type,source_fcg_type+'-'+str(claimID))
			name=source_fcg_type+'-'+str(claimID)+"_"+target_claim_type+str(claimID)
		else:
			source_fcg=create_weighted(p,rdf_path,model_path,graph_path,embed_path,source_claim_type,source_fcg_type)
			name=source_fcg_type+"_"+target_claim_type+str(claimID)
		nx.write_edgelist(source_fcg,os.path.join(writegraph_path,"{}.edgelist".format(name)))
		paths_of_interest_w[claimID]={}
		paths_of_interest_d[claimID]={}
		paths_of_interest_w[claimID]['target_claim']=chunkstring(target_claims[target_claims['claimID']==claimID]['claim_text'].values[0],100)
		paths_of_interest_d[claimID]['target_claim']=chunkstring(target_claims[target_claims['claimID']==claimID]['claim_text'].values[0],100)
		'''
		for every edge of interest, all edges connected to the source u and target v are reweighted by removing their log(degree)
		source_fcg2 acts as a temporary copy so that every edge, the graph can be reassigned to its original
		'''
		source_fcg2=source_fcg.copy()
		for edge in edges_of_interest[claimID]:
			u,v=edge
			######################Removing the log of source and target node from the incident edges
			for e in source_fcg.edges(u):
				if source_fcg.edges[e]['dist']<np.inf:
					diff=source_fcg.edges[e]['dist']*np.log10(source_fcg.degree(u))*0.5
					if source_fcg.edges[e]['weight']>diff:
						source_fcg.edges[e]['weight']=source_fcg.edges[e]['weight']-diff
					else:
						# print("Warning: diff>source weight")
						source_fcg.edges[e]['weight']=0
			for e in source_fcg.edges(v):
				if source_fcg.edges[e]['dist']<np.inf:
					diff=source_fcg.edges[e]['dist']*np.log10(source_fcg.degree(v))*0.5
					if source_fcg.edges[e]['weight']>diff:
						source_fcg.edges[e]['weight']=source_fcg.edges[e]['weight']-diff
					else:
						# print("Warning: diff>source weight")
						source_fcg.edges[e]['weight']=0
			###############################################################
			path_w=nx.shortest_path(source_fcg,source=u,target=v,weight='weight')
			path_d=nx.shortest_path(source_fcg,source=u,target=v,weight='dist')
			path_w_data={}
			path_d_data={}
			for i in range(len(path_w)-1):
				data=source_fcg.edges[path_w[i],path_w[i+1]]
				data['source_claim']=chunkstring(source_claims[source_claims['claimID']==data['claim_ID']]['claim_text'].values[0],75)
				path_w_data[str((path_w[i],path_w[i+1]))]=data
			for i in range(len(path_d)-1):
				data=source_fcg.edges[path_d[i],path_d[i+1]]
				data['source_claim']=chunkstring(source_claims[source_claims['claimID']==data['claim_ID']]['claim_text'].values[0],75)
				path_d_data[str((path_d[i],path_d[i+1]))]=data
			w=round(aggregate_edge_data(path_w_data,'weight'),2)
			d=round(aggregate_edge_data(path_w_data,'dist'),2)
			paths_of_interest_w[claimID][str((u,v,w,d))]=path_w_data
			w=round(aggregate_edge_data(path_d_data,'weight'),2)
			d=round(aggregate_edge_data(path_d_data,'dist'),2)
			paths_of_interest_d[claimID][str((u,v,w,d))]=path_d_data
			source_fcg=source_fcg2
		#storing the weighted and distance path files for obeservation
		with codecs.open(write_path+"_w.json","w","utf-8") as f:
			f.write(json.dumps(paths_of_interest_w,indent=5,ensure_ascii=False))
		with codecs.open(write_path+"_d.json","w","utf-8") as f:
			f.write(json.dumps(paths_of_interest_d,indent=5,ensure_ascii=False))
	create_ordered_paths(write_path,"w")
	create_ordered_paths(write_path,"d")

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Find shortest paths on co-cccurrence graphs')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/rdf_files/")
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/graphs/")
	parser.add_argument('-mp','--modelpath', metavar='model path',type=str,help='Model directory to load the model',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/models/claims-relatedness-model/claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27")
	parser.add_argument('-ep','--embedpath', metavar='embed path',type=str,help='Model directory to save and load embeddings',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/embeddings")
	parser.add_argument('-p','--passive',action='store_true',help='Passive or not',default=False)
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg','ffcg','ufcg','covid19'],help='True/False/Union/Covid19 FactCheckGraph')
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	args=parser.parse_args()
	# embed_claims(args.rdfpath,"roberta-base-nli-stsb-mean-tokens",args.embedpath,args.fcgtype)
	shortest_paths(args.rdfpath,args.modelpath,args.graphpath,args.embedpath,"tfcg",args.fcgtype,args.mode)