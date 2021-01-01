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
from statistics import mean,median
from collections import ChainMap
import sys
import csv

def cleanstring(string):
	string=re.sub(r'([a-z]*:)','',string)
	string=re.sub(r'([A-Z]*[a-z]+)_([A-Z]*[a-z]+)_',r'\1 \2 ',string)
	string=re.sub(r'([A-Z]*[a-z]+)_([A-Z]*[a-z]+)',r'\1 \2',string)
	string=re.sub(r'(_[0-9]*)','',string)
	string=re.sub(r'([A-Z][a-z]*)',r' \1',string)
	string=string.replace("( ","(").replace("  "," ").strip()
	return string

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

def clean_node_labels(node_label):
	node_label=node_label.split(':')[-1]
	#regex to find words in strings like "IllegalAllien" and "Want_3210203" and "President_of_the_United_States"
	regex_words=re.compile(r'([A-Z][a-z]*|[a-z]+)')
	node_label=" ".join(regex_words.findall(node_label))
	return node_label

def create_weighted(p,rdf_path,model_path,graph_path,graph_type,embed_path,claim_type,fcg_type,fcg_class):
	#Creating fcg read path
	if fcg_type not in set(['tfcg_co','ffcg_co','tfcg','ffcg']):
		fcg_parent_type=fcg_type.split('-')[0]
		fcg_path=os.path.join(graph_path,fcg_class,fcg_parent_type,"leave1out")
	else:
		fcg_path=os.path.join(graph_path,fcg_class,fcg_type)
	fcg=nx.read_edgelist(os.path.join(fcg_path,"{}.edgelist".format(fcg_type)),comments="@",create_using=eval(graph_type))
	temp_fcg=nx.DiGraph()
	temp_fcg.add_edges_from(list(fcg.edges.data()))
	#if graph is undirected, add a reverse edge to the directed graph, to simulate undirectedness
	if graph_type=='nx.MultiGraph':
		temp_fcg.add_edges_from(list(map(lambda x:(x[1],x[0],x[2]),list(fcg.edges.data()))))
	fcg=temp_fcg.copy()
	#Assigning the weight (dist*log(degree)) it again after the graph is pruned off its mutli edges
	for u,v,data in fcg.edges.data():
		dist=1
		weight=np.log2(fcg.degree(v))
		fcg.edges[u,v]['weight']=weight
		fcg.edges[u,v]['dist']=dist
	return fcg

def domb(numlist):
	domb=numlist[0]
	for j in numlist[1:]:
		if domb==0 and j==0:
			domb=0
		else:
			domb=(domb*j)/(domb+j-(domb*j))
	return domb

def aggregate_edge_data(evalues,mode):
	#mode can be dist or weight
	edgepair_weights=[]
	for edgepair,e2values in evalues.items():
		edgepair_weights.append(e2values[mode])
	return sum(edgepair_weights)

def aggregate_weights(claim_D,mode,mode2):
	#mode can be w, d or f
	#mode2 can be max, min, sum, mean
	edge_weights=[]
	for edge,evalues in claim_D.items():
		if type(evalues)!=list:
			u,v,w,d=eval(edge.replace("inf","np.inf"))
			edge_weights.append(eval(mode))
	return eval("{}(edge_weights)".format(mode2))

def create_ordered_paths(write_path,mode):
	#mode can be weight/dist
	rw_path=write_path+"_"+mode
	with codecs.open(rw_path+".json","r","utf-8") as f: 
		paths=json.loads(f.read())
	np.save(rw_path+"_claimIDs.npy",list(paths.keys()))
	for mode2 in ['sum','mean','max','median','domb']:
		ordered_paths={str((int(t[0]),aggregate_weights(t[1],mode,mode2))):t[1] for t in paths.items()}
		ordered_paths=sorted(ordered_paths.items(), key=lambda t: eval(t[0])[1],reverse=True)
		top10p=ordered_paths[:int(len(ordered_paths)*.10)]
		bot10p=ordered_paths[int(len(ordered_paths)*.90):]
		ordered_paths=OrderedDict(ordered_paths)
		with codecs.open(rw_path+"_"+mode2+".json","w","utf-8") as f:
			f.write(json.dumps(ordered_paths,indent=5,ensure_ascii=False))
		with codecs.open(rw_path+"_"+mode2+"_1.json","w","utf-8") as f:
			f.write(json.dumps(OrderedDict(top10p),indent=5,ensure_ascii=False))
		with codecs.open(rw_path+"_"+mode2+"_0.json","w","utf-8") as f:
			f.write(json.dumps(OrderedDict(bot10p),indent=5,ensure_ascii=False))

#Function to find node pairs in the source graph if they exist as edges in the target graph
def find_edges_of_interest(rdf_path,graph_path,graph_type,embed_path,source_fcg_type,target_fcg_type,fcg_class):
	suffix={"co_occur":"_co","fred":"_clean"}
	if target_fcg_type in source_fcg_type:
		fcg_type,target_claimID=source_fcg_type.split('-')
		claim_types={'tfcg_co':'true','ffcg_co':'false','tfcg':'true','ffcg':'false'}
		claim_type=claim_types[fcg_type]+"_claims"
		source_fcg_path=os.path.join(graph_path,fcg_class,fcg_type,"leave1out")
		target_fcg_path=os.path.join(rdf_path,claim_type)
		target_fcg_type="claim"+target_claimID+suffix[fcg_class]
	else:
		source_fcg_path=os.path.join(graph_path,fcg_class,source_fcg_type)
		target_fcg_path=os.path.join(graph_path,fcg_class,target_fcg_type)
	#loading source and target fcg
	source_fcg=nx.read_edgelist(os.path.join(source_fcg_path,"{}.edgelist".format(source_fcg_type)),comments="@",create_using=eval(graph_type))
	try:
		target_fcg=nx.read_edgelist(os.path.join(target_fcg_path,"{}.edgelist".format(target_fcg_type)),comments="@",create_using=eval(graph_type))
	except FileNotFoundError:
		target_fcg=eval(graph_type+'()')
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
		if u!=v:
			try:
				edges_of_interest[data['claim_ID']].add((u,v))
			except KeyError:
				edges_of_interest[data['claim_ID']]=set([(u,v)])
	return edges_of_interest


def find_paths_of_interest(index,rdf_path,graph_path,graph_type,embed_path,model_path,claimIDs,fcg_class,source_fcg_type,target_fcg_type,source_claim_type,target_claim_type,source_claims,target_claims,edges_of_interest,target_claims_embed,writegraph_path):
	paths_of_interest_w={}
	paths_of_interest_d={}
	#iterating throuhg the claimIDs of interest
	for claimID in claimIDs:
		if source_fcg_type==target_fcg_type:
			#checking if the source graph (without the given claim), has edges of interest. If not, we skip the claim .i.e continue in the for loop
			edges_of_interest=find_edges_of_interest(rdf_path,graph_path,graph_type,embed_path,source_fcg_type+'-'+str(claimID),target_fcg_type,fcg_class)
			if claimID not in set(list(edges_of_interest.keys())):
				continue
			claimIDs=list(edges_of_interest.keys())
		#getting index of claim
		claimIX=target_claims[target_claims['claimID']==claimID].index[0]
		p=np.array([target_claims_embed[claimIX]])
		#source_fcg is not a multigraph
		#if source and target are the same graph, then the graph without the given claim is fetched
		if source_fcg_type==target_fcg_type:
			source_fcg=create_weighted(p,rdf_path,model_path,graph_path,graph_type,embed_path,source_claim_type,source_fcg_type+'-'+str(claimID),fcg_class)
			name=source_fcg_type+'-'+str(claimID)+"_"+target_claim_type+str(claimID)
		else:
			source_fcg=create_weighted(p,rdf_path,model_path,graph_path,graph_type,embed_path,source_claim_type,source_fcg_type,fcg_class)
			name=source_fcg_type+"_"+target_claim_type+str(claimID)
		paths_of_interest_w[claimID]={}
		paths_of_interest_d[claimID]={}
		paths_of_interest_w[claimID]['target_claim']=chunkstring(target_claims[target_claims['claimID']==claimID]['claim_text'].values[0],100)
		paths_of_interest_d[claimID]['target_claim']=chunkstring(target_claims[target_claims['claimID']==claimID]['claim_text'].values[0],100)
		source_fcg2=source_fcg.copy()
		'''
		for every edge of interest, all edges connected to the source u and target v are reweighted by adding the source log(degree)
		'''
		z=len(edges_of_interest[claimID])
		zcount=0
		for edge in edges_of_interest[claimID]:
			u,v=edge
			if source_fcg.has_node(u) and source_fcg.has_node(v) and u!=v:
				'''
				If both node exists, we add the influence of source nodes i.e. the weight and log10(degree) to all the paths.
				We want to include source node weights to all parts starting from them so that:
				1. There are no 0 dist paths, even when they are directly connected
				2. While aggregating the entity pairs, we automatically include the weight of the an entity pair by not ignoring the weight of the source node
				'''
				for e in source_fcg.out_edges(u,data=True):
					data=e[2]
					#adding the influence of the log of degree of target node u
					data['weight']=data['dist']*(np.log2(source_fcg.degree(e[0]))+np.log2(source_fcg.degree(e[1])))
					source_fcg.edges[e[0],e[1]].update(data)
				for e in source_fcg.out_edges(v,data=True):
					data=e[2]
					#adding the influence of the log of degree of target node u
					data['weight']=data['dist']*(np.log2(source_fcg.degree(e[0]))+np.log2(source_fcg.degree(e[1])))
					source_fcg.edges[e[0],e[1]].update(data)
				#################################################################################
				if nx.has_path(source_fcg,u,v):
					#################################################################################
					path_d1=nx.shortest_path(source_fcg,source=u,target=v,weight='dist')
					path_d_data1={}
					path_d_formed_claim1=cleanstring(path_d1[0])
					#################################################################################
					for i in range(len(path_d1)-1):
						data=source_fcg.edges[path_d1[i],path_d1[i+1]]
						data['source_claim']=chunkstring(source_claims[source_claims['claimID']==data['claim_ID']]['claim_text'].values[0],75)
						path_d_data1[str((path_d1[i],path_d1[i+1]))]=data
						path_d_formed_claim1=path_d_formed_claim1+" "+cleanstring(path_d1[i+1])
					###########################################################################
					dw1=round(1/(1+aggregate_edge_data(path_d_data1,'weight')),3)
					dd1=round(1/(1+aggregate_edge_data(path_d_data1,'dist')),3)
					path_d_data1['formed_claim']=path_d_formed_claim1
					#################################################################################
					path_w1=nx.shortest_path(source_fcg,source=u,target=v,weight='weight')
					path_w_data1={}		
					path_w_formed_claim1=cleanstring(path_w1[0])
					#################################################################################
					for i in range(len(path_w1)-1):
						data=source_fcg.edges[path_w1[i],path_w1[i+1]]
						data['source_claim']=chunkstring(source_claims[source_claims['claimID']==data['claim_ID']]['claim_text'].values[0],75)
						path_w_data1[str((path_w1[i],path_w1[i+1]))]=data
						path_w_formed_claim1=path_w_formed_claim1+" "+cleanstring(path_w1[i+1])
					#################################################################################
					ww1=round(1/(1+aggregate_edge_data(path_w_data1,'weight')),3)
					wd1=round(1/(1+aggregate_edge_data(path_w_data1,'dist')),3)
					path_w_data1['formed_claim']=path_w_formed_claim1
				else:
					dw1=0
					dd1=0
					ww1=0
					wd1=0
					path_d_data1={}
					path_d_data1['formed_claim']=""
					path_w_data1={}
					path_w_data1['formed_claim']=""
				if nx.has_path(source_fcg,v,u):
					#################################################################################
					#################################################################################
					path_d2=nx.shortest_path(source_fcg,source=v,target=u,weight='dist')
					path_d_data2={}
					path_d_formed_claim2=cleanstring(path_d2[0])
					#################################################################################
					for i in range(len(path_d2)-1):
						data=source_fcg.edges[path_d2[i],path_d2[i+1]]
						data['source_claim']=chunkstring(source_claims[source_claims['claimID']==data['claim_ID']]['claim_text'].values[0],75)
						path_d_data2[str((path_d2[i],path_d2[i+1]))]=data
						path_d_formed_claim2=path_d_formed_claim2+" "+cleanstring(path_d2[i+1])
					#################################################################################
					dw2=round(1/(1+aggregate_edge_data(path_d_data2,'weight')),3)
					dd2=round(1/(1+aggregate_edge_data(path_d_data2,'dist')),3)
					path_d_data2['formed_claim']=path_d_formed_claim2
					#################################################################################
					#################################################################################
					path_w2=nx.shortest_path(source_fcg,source=v,target=u,weight='weight')
					path_w_data2={}
					path_w_formed_claim2=cleanstring(path_w2[0])
					#################################################################################
					for i in range(len(path_w2)-1):
						data=source_fcg.edges[path_w2[i],path_w2[i+1]]
						data['source_claim']=chunkstring(source_claims[source_claims['claimID']==data['claim_ID']]['claim_text'].values[0],75)
						path_w_data2[str((path_w2[i],path_w2[i+1]))]=data
						path_w_formed_claim2=path_w_formed_claim2+" "+cleanstring(path_w2[i+1])
					#################################################################################
					ww2=round(1/(1+aggregate_edge_data(path_w_data2,'weight')),3)
					wd2=round(1/(1+aggregate_edge_data(path_w_data2,'dist')),3)
					path_w_data2['formed_claim']=path_w_formed_claim2
				else:
					dw2=0
					dd2=0
					ww2=0
					wd2=0
					path_d_data2={}
					path_d_data2['formed_claim']=""
					path_w_data2={}
					path_w_data2['formed_claim']=""
				if dd1>dd2:
					paths_of_interest_d[claimID][str((u,v,dw1,dd1))]=path_d_data1
				else:
					paths_of_interest_d[claimID][str((u,v,dw2,dd2))]=path_d_data2
				if ww1>ww2:
					paths_of_interest_w[claimID][str((u,v,ww1,wd1))]=path_w_data1
				else:
					paths_of_interest_w[claimID][str((u,v,ww2,wd2))]=path_w_data2
			else:
				path_data={}
				path_data['formed_claim']=""
				paths_of_interest_d[claimID][str((u,v,0,0))]=path_data
				paths_of_interest_w[claimID][str((u,v,0,0))]=path_data
			source_fcg=source_fcg2
	return index,paths_of_interest_w,paths_of_interest_d
'''
Function does the following
1. Finds node pairs of interest from graph built through target claims by using the function find_edges_of_interest
2. Creates a weighted source graph for each target claim using the function create_weighted
3. Finds shortest path for each edge of interest in the target graph
'''
def find_shortest_paths(rdf_path,model_path,graph_path,graph_type,embed_path,source_fcg_type,target_fcg_type,fcg_class,cpu):
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
	target_claims_embed=pd.read_csv(os.path.join(embed_path,target_claim_type+"_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	#limiting claimIDs to edges of interest if both graphs are different
	if source_fcg_type==target_fcg_type:
		edges_of_interest={}
		claimIDs=source_claims['claimID'].tolist()
	else:
		edges_of_interest=find_edges_of_interest(rdf_path,graph_path,graph_type,embed_path,source_fcg_type,target_fcg_type,fcg_class)
		claimIDs=list(edges_of_interest.keys())
	#finding paths of interest
	n=int(len(claimIDs)/cpu)+1
	if cpu>1:
		pool=mp.Pool(processes=cpu)
		results=[pool.apply_async(find_paths_of_interest, args=(index,rdf_path,graph_path,graph_type,embed_path,model_path,claimIDs[index*n:(index+1)*n],fcg_class,source_fcg_type,target_fcg_type,source_claim_type,target_claim_type,source_claims,target_claims,edges_of_interest,target_claims_embed,writegraph_path)) for index in range(cpu)]
		output=sorted([p.get() for p in results],key=lambda x:x[0])
		paths_of_interest_w=dict(ChainMap(*map(lambda x:x[1],output)))
		paths_of_interest_d=dict(ChainMap(*map(lambda x:x[2],output)))
	else:
		index,paths_of_interest_w,paths_of_interest_d=find_paths_of_interest(0,rdf_path,graph_path,graph_type,embed_path,model_path,claimIDs[0:n],fcg_class,source_fcg_type,target_fcg_type,source_claim_type,target_claim_type,source_claims,target_claims,edges_of_interest,target_claims_embed,writegraph_path)
	graph_types={'nx.MultiGraph':'undirected','nx.MultiDiGraph':'directed'}
	graph_type=graph_types[graph_type]
	#storing the weighted and distance path files for observation
	write_path=write_path+"_"+graph_type
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
	parser.add_argument('-st','--sfcgtype', metavar='Source FactCheckGraph type',type=str,choices=['tfcg','ffcg','tfcg_co','ffcg_co','ufcg','covid19'],help='True/False/Union/Covid19 FactCheckGraph')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg','ffcg','tfcg_co','ffcg_co','ufcg','covid19'],help='True/False/Union/Covid19 FactCheckGraph')
	parser.add_argument('-fc','--fcgclass', metavar='FactCheckGraph class',type=str,choices=['co_occur','fred'])
	parser.add_argument('-gt','--graphtype', metavar='Graph Type Directed/Undirected',type=str,choices=['directed','undirected'])
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	graph_types={'undirected':'nx.MultiGraph','directed':'nx.MultiDiGraph'}
	args=parser.parse_args()
	find_shortest_paths(args.rdfpath,args.modelpath,args.graphpath,graph_types[args.graphtype],args.embedpath,args.sfcgtype,args.fcgtype,args.fcgclass,args.cpu)