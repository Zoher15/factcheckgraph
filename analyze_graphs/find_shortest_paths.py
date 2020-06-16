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
from collections import ChainMap
import sys
# sys.path.insert(1, '/geode2/home/u110/zkachwal/BigRed3/factcheckgraph/create_graphs')
# from create_fred import *
# from create_co_occur import *
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
	with open(os.path.join(embed_path,claim_type+'_claims_embeddings_({}).tsv'.format(model_path.split("/")[-1])),'w',newline='') as f:
		for vector in claims_embeddings:
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)
	with open(os.path.join(embed_path,claim_type+'_claims_embeddings_labels_({}).tsv'.format(model_path.split("/")[-1])),'w',newline='') as f:
		vector=['claim_text','claimID','rating']
		tsv_output=csv.writer(f,delimiter='\t')
		tsv_output.writerow(vector)
		for i in range(len(claims)):
			vector=list((claims.iloc[i]['claim_text'],claims.iloc[i]['claimID'],claim_type))
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)

def clean_node_labels(node_label):
	node_label=node_label.split(':')[-1]
	#regex to find words in strings like "IllegalAllien" and "Want_3210203" and "President_of_the_United_States"
	regex_words=re.compile(r'([A-Z][a-z]*|[a-z]+)')
	node_label=" ".join(regex_words.findall(node_label))
	return node_label

def embed_nodes(graph_path,model_path,embed_path,fcg_type,fcg_class):
	os.makedirs(embed_path, exist_ok=True)
	model = SentenceTransformer(model_path)
	fcg=nx.read_edgelist(os.path.join(graph_path,fcg_class,fcg_type,"{}.edgelist".format(fcg_type)),comments="@",create_using=nx.MultiGraph)
	nodes_list=list(fcg.nodes)
	nodes_clean_list=list(map(clean_node_labels,nodes_list))
	nodes_embeddings=model.encode(nodes_clean_list)
	with open(os.path.join(embed_path,fcg_type+'_nodes_embeddings_({}).tsv'.format(model_path.split("/")[-1])),'w',newline='') as f:
		for vector in nodes_embeddings:
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)
	with open(os.path.join(embed_path,fcg_type+'_nodes_embeddings_labels_({}).tsv'.format(model_path.split("/")[-1])),'w',newline='') as f:
		vector=['node_label','node_clean_label']
		tsv_output=csv.writer(f,delimiter='\t')
		tsv_output.writerow(vector)
		for i in range(len(nodes_list)):
			vector=list((nodes_list[i],nodes_clean_list[i]))
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)

def create_weighted(p,rdf_path,model_path,graph_path,embed_path,claim_type,fcg_type,fcg_class):
	#Creating fcg read path
	if fcg_type not in set(['tfcg_co','ffcg_co','tfcg','ffcg']):
		fcg_parent_type=fcg_type.split('-')[0]
		fcg_path=os.path.join(graph_path,fcg_class,fcg_parent_type,"leave1out")
	else:
		fcg_path=os.path.join(graph_path,fcg_class,fcg_type)
	#Adding weights to edges
	if fcg_class=='co_occur':
		claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
		claim_IDs=claims['claimID'].tolist()
		embed=pd.read_csv(os.path.join(embed_path,claim_type+"_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	elif fcg_class=='fred':
		embed=pd.read_csv(os.path.join(embed_path,fcg_type.split('-')[0]+"_nodes_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
		labels=pd.read_csv(os.path.join(embed_path,fcg_type.split('-')[0]+"_nodes_embeddings_labels_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t")
	#cosine similarity
	simil_p=cosine_similarity(p,embed)[0]
	#normalized angular distance
	dist_p=np.arccos(simil_p)/np.pi
	fcg=nx.read_edgelist(os.path.join(fcg_path,"{}.edgelist".format(fcg_type)),comments="@",create_using=nx.MultiGraph)
	temp_fcg=nx.MultiDiGraph()
	temp_fcg.add_edges_from(list(fcg.edges.data(keys=True)))
	temp_fcg.add_edges_from(list(map(lambda x:(x[1],x[0],x[2],x[3]),list(fcg.edges.data(keys=True)))))
	fcg=temp_fcg.copy()
	#assigning weight by summing the log of adjacent nodes, and dividing by the similiarity of the claim with the target predicate
	for u,v,k,d in fcg.edges.data(keys=True):
		if fcg_class=='co_occur':
			claimID=d['claim_ID']
			claimIX=claims[claims['claimID']==claimID].index[0]
			# uw=np.log10(fcg.degree(u))
			vw=np.log10(fcg.degree(v))
			dist=dist_p[claimIX]
			weight=dist*(vw)#+uw)*0.5
		elif fcg_class=='fred':
			# uIX=labels[labels['node_label']==u].index[0]
			vIX=labels[labels['node_label']==v].index[0]
			# uw=np.log10(fcg.degree(u))
			vw=np.log10(fcg.degree(v))
			dist=dist_p[vIX]#+dist_p[uIX]
			weight=dist*(vw)#+uw)*0.5
	fcg.edges[u,v,k]['dist']=dist
	fcg.edges[u,v,k]['weight']=weight
	#Removing multiedges, by selecting the shortest one
	fcg2=nx.DiGraph()
	for u,v,k,data in fcg.edges.data(keys=True):
		if not fcg2.has_edge(u,v):
			fcg2.add_edge(u,v)
			if len(fcg.get_edge_data(u,v))>1:
				datalist=fcg.get_edge_data(u,v)
				#finding the data dict among all the multiedges between u and v with the min distance
				data=min(fcg.get_edge_data(u,v).items(), key=lambda x:x[1]['dist'])[1]
			fcg2.edges[u,v].update(data)
	return fcg2

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
def find_edges_of_interest(rdf_path,graph_path,embed_path,source_fcg_type,target_fcg_type,fcg_class):
	suffix={"co_occur":"_co","fred":"_co"}
	if target_fcg_type in source_fcg_type:
		fcg_type,target_claimID=source_fcg_type.split('-')
		claim_types={'tfcg_co':'true','ffcg_co':'false','tfcg':'true','ffcg':'false'}
		claim_type=claim_types[fcg_type]+"_claims"
		source_fcg_path=os.path.join(graph_path,fcg_class,fcg_type,"leave1out")
		target_fcg_path=os.path.join(rdf_path,claim_type)
		target_fcg_type="claim"+target_claimID+suffix[fcg_class]
	else:
		target_fcg_types={"ffcg_co":"ffcg_co","ffcg":"ffcg_co"}
		target_fcg_type=target_fcg_types[target_fcg_type]
		source_fcg_path=os.path.join(graph_path,fcg_class,source_fcg_type)
		target_fcg_path=os.path.join(graph_path,"co_occur",target_fcg_type)
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
	###################################################################################
	edges=target_fcg.edges.data(keys=True)
	for edge in edges:
		u,v,k,data=edge
		if source_fcg.has_node(u) and source_fcg.has_node(v) and nx.has_path(source_fcg,u,v):#and not source_fcg.has_edge(u,v) and 
			try:
				edges_of_interest[data['claim_ID']].add((u,v))
			except KeyError:
				edges_of_interest[data['claim_ID']]=set([(u,v)])
	return edges_of_interest


def find_paths_of_interest(index,rdf_path,graph_path,embed_path,model_path,claimIDs,fcg_class,source_fcg_type,target_fcg_type,source_claim_type,target_claim_type,source_claims,target_claims,edges_of_interest,target_claims_embed,writegraph_path):
	paths_of_interest_w={}
	paths_of_interest_d={}
	#iterating throuhg the claimIDs of interest
	for claimID in claimIDs:
		if source_fcg_type==target_fcg_type:
			#checking if the source graph (without the given claim), has edges of interest. If not, we skip the claim .i.e continue in the for loop
			edges_of_interest=find_edges_of_interest(rdf_path,graph_path,embed_path,source_fcg_type+'-'+str(claimID),target_fcg_type,fcg_class)
			if claimID not in set(list(edges_of_interest.keys())):
				continue
		#getting index of claim
		claimIX=target_claims[target_claims['claimID']==claimID].index[0]
		p=np.array([target_claims_embed[claimIX]])
		#source_fcg is not a multigraph
		#if source and target are the same graph, then the graph without the given claim is fetched
		if source_fcg_type==target_fcg_type:
			source_fcg=create_weighted(p,rdf_path,model_path,graph_path,embed_path,source_claim_type,source_fcg_type+'-'+str(claimID),fcg_class)
			name=source_fcg_type+'-'+str(claimID)+"_"+target_claim_type+str(claimID)
		else:
			source_fcg=create_weighted(p,rdf_path,model_path,graph_path,embed_path,source_claim_type,source_fcg_type,fcg_class)
			name=source_fcg_type+"_"+target_claim_type+str(claimID)
		paths_of_interest_w[claimID]={}
		paths_of_interest_d[claimID]={}
		paths_of_interest_w[claimID]['target_claim']=chunkstring(target_claims[target_claims['claimID']==claimID]['claim_text'].values[0],100)
		paths_of_interest_d[claimID]['target_claim']=chunkstring(target_claims[target_claims['claimID']==claimID]['claim_text'].values[0],100)
		'''
		for every edge of interest, all edges connected to the source u and target v are reweighted by removing their log(degree)
		'''
		for edge in edges_of_interest[claimID]:
			u,v=edge
			if fcg_class=='co_occur':
				for e in source_fcg.in_edges(v,data=True):
					data=e[2]
					#removing the influence of the log of degree of target node v
					data['weight']=data['dist']
					source_fcg.edges[e[0],e[1]].update(data)
			elif fcg_class=='fred':
				for e in source_fcg.in_edges(v,data=True):
					data=e[2]
					#removing the influence of the log of degree and the distance of target node v
					data['dist']=0
					data['weight']=0
					source_fcg.edges[e[0],e[1]].update(data)
			###############################################################
			path_w=nx.shortest_path(source_fcg,source=u,target=v,weight='weight')
			path_d=nx.shortest_path(source_fcg,source=u,target=v,weight='dist')
			path_w_data={}
			path_d_data={}
			for i in range(len(path_w)-1):
				data=source_fcg.edges[path_w[i],path_w[i+1]]
				data['source_claim']=chunkstring(source_claims[source_claims['claimID']==data['claim_ID']]['claim_text'].values[0],75)
				path_w_data[str((path_w[i],path_w[i+1]))]=data
			###################################################################
			for i in range(len(path_d)-1):
				data=source_fcg.edges[path_d[i],path_d[i+1]]
				data['source_claim']=chunkstring(source_claims[source_claims['claimID']==data['claim_ID']]['claim_text'].values[0],75)
				path_d_data[str((path_d[i],path_d[i+1]))]=data
			###################################################################
			w=round(aggregate_edge_data(path_w_data,'weight'),3)
			d=round(aggregate_edge_data(path_w_data,'dist'),3)
			paths_of_interest_w[claimID][str((u,v,w,d))]=path_w_data
			w=round(aggregate_edge_data(path_d_data,'weight'),3)
			d=round(aggregate_edge_data(path_d_data,'dist'),3)
			paths_of_interest_d[claimID][str((u,v,w,d))]=path_d_data
			# source_fcg=source_fcg2
	return index,paths_of_interest_w,paths_of_interest_d
'''
Function does the following
1. Finds node pairs of interest from graph built through target claims by using the function find_edges_of_interest
2. Creates a weighted source graph for each target claim using the function create_weighted
3. Finds shortest path for each edge of interest in the target graph
'''
def find_shortest_paths(rdf_path,model_path,graph_path,embed_path,source_fcg_type,target_fcg_type,fcg_class,cpu):
	fcg_types={'co_occur':{'tfcg':'tfcg_co','ffcg':'ffcg_co'},'fred':{'tfcg':'tfcg','ffcg':'ffcg'}}
	#setting source fcg_type
	source_fcg_type=fcg_types[fcg_class][source_fcg_type]
	#setting target fcg_type
	target_fcg_type=fcg_types[fcg_class][target_fcg_type]
	claim_types={'tfcg_co':'true','ffcg_co':'false','tfcg':'true','ffcg':'false'}
	#setting source claim_type
	source_claim_type=claim_types[source_fcg_type]
	#setting target claim_type
	target_claim_type=claim_types[target_fcg_type]
	#path to store the shortest paths
	write_path=os.path.join(graph_path,fcg_class,"paths",source_fcg_type+"_"+target_claim_type+"_({})".format(model_path.split("/")[-1]))
	os.makedirs(write_path,exist_ok=True)
	write_path=os.path.join(write_path,"paths")
	#path to store the weghted graphs
	writegraph_path=os.path.join(graph_path,"weighted","graphs_"+fcg_class+"_"+source_fcg_type+"_"+target_claim_type+"_({})".format(model_path.split("/")[-1]))
	os.makedirs(writegraph_path,exist_ok=True)
	#path for the graph that is being used to fact-check
	source_fcg_path=os.path.join(graph_path,fcg_class,source_fcg_type)
	#loading claims
	source_claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(source_claim_type)))
	target_claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(target_claim_type)))
	#loading target claim embeddings
	try:
		target_claims_embed=pd.read_csv(os.path.join(embed_path,target_claim_type+"_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	except FileNotFoundError:
		embed_claims(rdf_path,model_path,embed_path,target_claim_type)
		target_claims_embed=pd.read_csv(os.path.join(embed_path,target_claim_type+"_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	#limiting claimIDs to edges of interest if both graphs are different
	if source_fcg_type==target_fcg_type:
		edges_of_interest={}
		claimIDs=source_claims['claimID'].tolist()
	else:
		edges_of_interest=find_edges_of_interest(rdf_path,graph_path,embed_path,source_fcg_type,target_fcg_type,fcg_class)
		claimIDs=list(edges_of_interest.keys())
	#finding paths of interest
	n=int(len(claimIDs)/cpu)+1
	if cpu>1:
		pool=mp.Pool(processes=cpu)
		results=[pool.apply_async(find_paths_of_interest, args=(index,rdf_path,graph_path,embed_path,model_path,claimIDs[index*n:(index+1)*n],fcg_class,source_fcg_type,target_fcg_type,source_claim_type,target_claim_type,source_claims,target_claims,edges_of_interest,target_claims_embed,writegraph_path)) for index in range(cpu)]
		output=sorted([p.get() for p in results],key=lambda x:x[0])
		paths_of_interest_w=dict(ChainMap(*map(lambda x:x[1],output)))
		paths_of_interest_d=dict(ChainMap(*map(lambda x:x[2],output)))
	else:
		index,paths_of_interest_w,paths_of_interest_d=find_paths_of_interest(0,rdf_path,graph_path,embed_path,model_path,claimIDs[0:n],fcg_class,source_fcg_type,target_fcg_type,source_claim_type,target_claim_type,source_claims,target_claims,edges_of_interest,target_claims_embed,writegraph_path)
	#storing the weighted and distance path files for observation
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
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg','ffcg','tfcg_co','ffcg_co','ufcg','covid19'],help='True/False/Union/Covid19 FactCheckGraph')
	parser.add_argument('-fc','--fcgclass', metavar='FactCheckGraph class',type=str,choices=['co_occur','fred'])
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	args=parser.parse_args()
	# embed_claims(args.rdfpath,"roberta-base-nli-stsb-mean-tokens",args.embedpath,args.fcgtype)
	find_shortest_paths(args.rdfpath,args.modelpath,args.graphpath,args.embedpath,"tfcg",args.fcgtype,args.fcgclass,args.cpu)