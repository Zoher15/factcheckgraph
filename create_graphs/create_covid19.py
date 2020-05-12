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
from collections import OrderedDict
from statistics import mean
import csv

def create_fred(rdf_path,claim_type,passive,cpu):
	claims_path=os.path.join(rdf_path,"{}_claims".format(claim_type))
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
	claim_IDs=claims['claimID'].tolist()
	np.save(os.path.join(rdf_path,"{}_claimID.noy".format(claim_type)),claim_IDs)
	init=900
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
	model = SentenceTransformer(model_path)
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
	claims_list=list(claims['claim_text'])
	claims_embeddings=model.encode(claims_list)
	with open(os.path.join(embed_path,'{}_claims_embeddings.tsv'.format(claim_type)),'w',newline='') as f:
		for vector in claims_embeddings:
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)
	with open(os.path.join(embed_path,'{}_claims_embeddings_labels.tsv'.format(claim_type)),'w',newline='') as f:
		vector=['claim_text','claimID']#,'rating']
		tsv_output=csv.writer(f,delimiter='\t')
		tsv_output.writerow(vector)
		for i in range(len(claims)):
			vector=list((claims.iloc[i]['claim_text'],claims.iloc[i]['claimID']))#,claims.iloc[i]['factcheck_rating']))
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)

def create_weighted(p,rdf_path,graph_path,embed_path,claim_type):
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
	claim_IDs=claims['claimID'].tolist()
	claims_embed=pd.read_csv(os.path.join(embed_path,"{}_claims_embeddings.tsv".format(claim_type)),delimiter="\t",header=None).values
	simil_p=cosine_similarity(p,claims_embed)[0]
	#no negative similarity
	simil_p[simil_p<0]=0
	fcg_path=os.path.join(graph_path,"co-occur",claim_type)
	fcg_co=nx.read_edgelist(os.path.join(fcg_path,"{}.edgelist".format(claim_type)),comments="@",create_using=nx.MultiGraph)
	#assigning weight by summing the log of adjacent nodes, and dividing by the similiarity of the claim with the target predicate
	for u,v,k,claimID in fcg_co.edges.data('claim_ID',keys=True):
		claimIX=claims[claims['claimID']==claimID].index[0]
		uw=np.log2(fcg_co.degree(u))
		vw=np.log2(fcg_co.degree(v))
		if simil_p[claimIX]>0:
			dist=float(1)/simil_p[claimIX]
			weight=(uw+vw)*dist
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

def create_ordered_paths(graph_path,mode):
	#mode can be weight/dist
	rw_path=os.path.join(graph_path,"co-occur","paths","paths_{}".format(mode))
	with codecs.open(rw_path+".json","r","utf-8") as f: 
		paths_w=json.loads(f.read())
	for mode2 in ['sum','mean','max','min']:
		ordered_paths=OrderedDict(sorted(paths_w.items(), key=lambda t: aggregate_weights(t[1],mode2,mode)))
		with codecs.open(rw_path+"_{}.json".format(mode2),"w","utf-8") as f:
			f.write(json.dumps(ordered_paths,indent=5,ensure_ascii=False))

def source_target(rdf_path,graph_path,embed_path,source_claim_type,target_claim_type):
	source_fcg_path=os.path.join(graph_path,"co-occur",source_claim_type)
	source_path=os.path.join(graph_path,"co-occur",source_claim_type)
	source_fcg=nx.read_edgelist(os.path.join(source_fcg_path,"{}.edgelist".format(source_claim_type)),comments="@",create_using=nx.MultiGraph)
	target_fcg_path=os.path.join(graph_path,"co-occur",target_claim_type)
	target_path=os.path.join(graph_path,"co-occur",target_claim_type)
	target_fcg=nx.read_edgelist(os.path.join(target_fcg_path,"{}.edgelist".format(target_claim_type)),comments="@",create_using=nx.MultiGraph)
	edges_of_interest={}
	pairs_of_interest=[]
	for edge in target_fcg.edges.data(keys=True):
		u,v,k,data=edge
		if not source_fcg.has_edge(u,v) and (u,v) not in pairs_of_interest and source_fcg.has_node(u) and source_fcg.has_node(v) and nx.has_path(source_fcg,u,v):
			try:
				edges_of_interest[data['claim_ID']].append((u,v))
			except KeyError:
				edges_of_interest[data['claim_ID']]=[(u,v)]
			pairs_of_interest.append((u,v))
	return pairs_of_interest,edges_of_interest

def shortest_paths(rdf_path,graph_path,embed_path,source_claim_type,target_claim_type):
	write_path=os.path.join(graph_path,"co-occur","paths","paths")
	source_fcg_path=os.path.join(graph_path,"co-occur",source_claim_type)
	pairs_of_interest,edges_of_interest=source_target(rdf_path,graph_path,embed_path,source_claim_type,target_claim_type)
	#loading existing files
	try:
		with codecs.open(write_path+"_w.json","r","utf-8") as f:
			paths_of_interest_w=json.loads(f.read())
	except FileNotFoundError:
		paths_of_interest_w={}
	try:
		with codecs.open(write_path+"_d.json","r","utf-8") as f:
			paths_of_interest_s=json.loads(f.read())
	except FileNotFoundError:
		paths_of_interest_s={}
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(target_claim_type)))
	source_claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(source_claim_type)))
	claims_embed=pd.read_csv(os.path.join(embed_path,"{}_claims_embeddings.tsv".format(target_claim_type)),delimiter="\t",header=None).values
	for claimID in list(edges_of_interest.keys()):
		claimIX=claims[claims['claimID']==claimID].index[0]
		p=np.array([claims_embed[claimIX]])
		source_fcg=create_weighted(p,rdf_path,graph_path,embed_path,source_claim_type)
		source_fcg2=source_fcg.copy()
		name=source_claim_type+"_"+str(claimID)
		nx.write_edgelist(source_fcg,os.path.join(source_fcg_path,"{}.edgelist".format(name)))
		paths_of_interest_w[claimID]={}
		paths_of_interest_s[claimID]={}
		paths_of_interest_w[claimID]['target_claim']=chunkstring(claims[claims['claimID']==claimID]['claim_text'].values[0],100)
		paths_of_interest_s[claimID]['target_claim']=chunkstring(claims[claims['claimID']==claimID]['claim_text'].values[0],100)
		for edge in edges_of_interest[claimID]:
			u,v=edge
			######################Removing the log of source and target node from the incident edges
			for e in source_fcg.edges(u):
				if source_fcg.edges[e]['dist']<np.inf:
					diff=source_fcg.edges[e]['dist']*np.log2(source_fcg.degree(u))
					if source_fcg.edges[e]['weight']>diff:
						source_fcg.edges[e]['weight']=source_fcg.edges[e]['weight']-diff
					else:
						source_fcg.edges[e]['weight']=0
			for e in source_fcg.edges(v):
				if source_fcg.edges[e]['dist']<np.inf:
					diff=source_fcg.edges[e]['dist']*np.log2(source_fcg.degree(v))
					if source_fcg.edges[e]['weight']>diff:
						source_fcg.edges[e]['weight']=source_fcg.edges[e]['weight']-diff
					else:
						source_fcg.edges[e]['weight']=0
			###############################################################
			path_w=nx.shortest_path(source_fcg,source=u,target=v,weight='weight')
			path_s=nx.shortest_path(source_fcg,source=u,target=v,weight='dist')
			path_w_data={}
			path_s_data={}
			for i in range(len(path_w)-1):
				data=source_fcg.edges[path_w[i],path_w[i+1]]
				data['source_claim']=chunkstring(source_claims[source_claims['claimID']==data['claim_ID']]['claim_text'].values[0],75)
				path_w_data[str((path_w[i],path_w[i+1]))]=data
			for i in range(len(path_s)-1):
				data=source_fcg.edges[path_s[i],path_s[i+1]]
				data['source_claim']=chunkstring(source_claims[source_claims['claimID']==data['claim_ID']]['claim_text'].values[0],75)
				path_s_data[str((path_s[i],path_s[i+1]))]=data
			w=round(aggregate_edge_data(path_w_data,'weight'),2)
			d=round(aggregate_edge_data(path_w_data,'dist'),2)
			paths_of_interest_w[claimID][str((u,v,w,d))]=path_w_data
			w=round(aggregate_edge_data(path_s_data,'weight'),2)
			d=round(aggregate_edge_data(path_s_data,'dist'),2)
			paths_of_interest_s[claimID][str((u,v,w,d))]=path_s_data
			source_fcg=source_fcg2.copy()
		with codecs.open(write_path+"_w.json","w","utf-8") as f:
			f.write(json.dumps(paths_of_interest_w,indent=5,ensure_ascii=False))
		with codecs.open(write_path+"_d.json","w","utf-8") as f:
			f.write(json.dumps(paths_of_interest_s,indent=5,ensure_ascii=False))

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
	# create_fred_target(args.rdfpath,args.embedpath,"tweet")
	# create_fred(args.rdfpath,args.fcgtype,args.passive,args.cpu)
	# create_co_occur(args.rdfpath,args.graphpath,args.fcgtype)
	# embed_claims(args.rdfpath,args.modelpath,args.embedpath,args.fcgtype)
	# embed_claims(args.rdfpath,"roberta-base-nli-stsb-mean-tokens",args.embedpath,"covid19")
	# embed_claims(args.rdfpath,"roberta-base-nli-stsb-mean-tokens",args.embedpath,"covid19topics")
	shortest_paths(args.rdfpath,args.graphpath,args.embedpath,"covid19","covid19topics")
	create_ordered_paths(args.graphpath,"w")
	create_ordered_paths(args.graphpath,"d")