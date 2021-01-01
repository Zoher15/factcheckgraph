import os
import re
import sys
import time
import argparse
import random
import networkx as nx
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import codecs
import json
import numpy as np
import html
from operator import itemgetter
from collections import ChainMap 
from itertools import chain
import multiprocessing as mp
from pprint import pprint
import re

def nodelabel_mapper(text):
	regex_vn=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/vn\/data\/([a-zA-Z]*)_.*')
	regex_dbpedia=re.compile(r'^http:\/\/dbpedia\.org\/resource\/(.*)')
	regex_quant=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/quantifiers\.owl#.*')
	regex_schema=re.compile(r'^http:\/\/schema\.org.*')
	regex_fred=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#([a-zA-Z]*)_.*')
	regex_fredup=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#([A-Z]+[a-zA-Z]*)')
	try:
		d={
		bool(regex_vn.match(text)):'vn:'+text.split("/")[-1].split("#")[-1],
		bool(regex_dbpedia.match(text)):'db:'+text.split("/")[-1].split("#")[-1],
		bool(regex_quant.match(text)):'quant:'+text.split("/")[-1].split("#")[-1],
		bool(regex_schema.match(text)):'schema:'+text.split("/")[-1].split("#")[-1],
		bool(regex_fred.match(text)):'fred:'+text.split("/")[-1].split("#")[-1],
		bool(regex_fredup.match(text)):'fu:'+text.split("/")[-1].split("#")[-1],
		}
		return d[True]
	except KeyError:
		return 'un:'+text.split("/")[-1].split("#")[-1]

#Function save individual claim graphs
def saveClaimGraph(claim_g,filename,graph_type):
	nx.write_edgelist(claim_g,filename+"_clean.edgelist")
	claim_g=nx.read_edgelist(filename+"_clean.edgelist",comments="@")
	nx.write_graphml(claim_g,filename+"_clean.graphml",prettyprint=True)
	plotFredGraph(claim_g,filename+"_clean")

#Function to plot a networkx graph
def plotFredGraph(claim_g,filename):
	plt.figure()
	pos=nx.spring_layout(claim_g)
	nx.draw_networkx(claim_g,pos,with_labels=True,node_size=400)
	edge_labels={(edge[0], edge[1]): edge[2]['label'] for edge in claim_g.edges(data=True)}
	nx.draw_networkx_edge_labels(claim_g,pos,edge_labels=edge_labels)
	plt.axis('off')
	plt.savefig(filename)
	plt.close()
	plt.clf()

def contractClaimGraph(claim_g,contract_edgelist):
	#edge contraction needs to be done in a bfs fashion using topological sort
	#creating a temporary bfs graph
	temp_g=nx.DiGraph()
	temp_g.add_edges_from(contract_edgelist)
	#topological sorted order of nodes
	while True:
		try:
			tsnodes=list(nx.topological_sort(temp_g))
			break
		except:
			print("#####################################")
			print("Cycle Found!")
			cycle_nodes=set([])
			for edge in list(nx.find_cycle(temp_g,orientation='original')):
				cycle_nodes.add(edge[0])
				cycle_nodes.add(edge[1])
			degree_dict=dict(claim_g.degree(cycle_nodes))
			print("Degree of cycle nodes\n",degree_dict)
			min_key=min(degree_dict, key=degree_dict.get)
			print("Node removed:",min_key)
			temp_g.remove_node(min_key)
	#contractings edges from the leave nodes to the root
	for nodes in list(nx.edge_bfs(temp_g,tsnodes))[::-1]:
		if claim_g.has_node(nodes[0]) and claim_g.has_node(nodes[1]):
			claim_g=nx.contracted_nodes(claim_g,nodes[0],nodes[1],self_loops=False)
	return claim_g


#Function to return a clean graph, depending on the edges to delete and contract
def cleanClaimGraph(claim_g,clean_claims):
	nodes2remove=clean_claims['nodes2remove']
	nodes2contract=clean_claims['nodes2contract']
	contract_edgelist=[edge for edgelist in nodes2contract.values() for edge in edgelist]
	remove_nodelist=[node for nodelist in nodes2remove.values() for node in nodelist]
	claim_g=contractClaimGraph(claim_g,sorted(contract_edgelist))
	'''
	After contracting edges, sometimes the edge hasQuality now exists between two words like Related and RelatedAccount.
	This edge  Related-{hasQuality}-RelatedAccount is a result of the base contraction. This edge should be contracted.
	Because this edge-to-be-contracted will not be detected by the function checkClaimGraph, we should do the contraction manually.
	'''
	contract_edgelist=[]
	edgelist=list(claim_g.edges(data=True))
	regex_fred=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#([a-zA-Z]*)_.*')
	regex_vn=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/vn\/data\/([a-zA-Z]*)_.*')
	regex_dbpedia=re.compile(r'^http:\/\/dbpedia\.org\/resource\/(.*)')
	for u,v,d in edgelist:
		if d['label']=='associatedWith' or d['label']=='hasQuality' or d['label']=='sameAs' or d['label']=='equivalentClass' or d['label']=='type' or d['label']=='subClassOf':
			if u.split("/")[-1].split("#")[-1].lower() in v.split("/")[-1].split("#")[-1].lower():
				if (regex_dbpedia.match(v) and not regex_dbpedia.match(u)) or (regex_vn.match(v) and not regex_vn.match(u)):
					contract_edgelist.append((v,u))
					print((nodelabel_mapper(v),nodelabel_mapper(u),d['label']))
				else:
					contract_edgelist.append((u,v))
					print((nodelabel_mapper(u),nodelabel_mapper(v),d['label']))
			elif v.split("/")[-1].split("#")[-1].lower() in u.split("/")[-1].split("#")[-1].lower():
				if (regex_dbpedia.match(u) and not regex_dbpedia.match(v)) or (regex_vn.match(u) and not regex_vn.match(v)):
					contract_edgelist.append((u,v))
					print((nodelabel_mapper(u),nodelabel_mapper(v),d['label']))
				else:
					contract_edgelist.append((v,u))
					print((nodelabel_mapper(v),nodelabel_mapper(u),d['label']))
	claim_g=contractClaimGraph(claim_g,sorted(contract_edgelist))
	#remove nodes
	for node in remove_nodelist:
		if claim_g.has_node(node):
			claim_g.remove_node(node)
	#removing isolates
	claim_g.remove_nodes_from(list(nx.isolates(claim_g)))
	situation_node='http://www.ontologydesignpatterns.org/ont/fred/domain.owl#Situation'
	if claim_g.has_node(situation_node):
		claim_g.remove_node(situation_node)
	return claim_g

#Function to stitch/compile graphs in an iterative way. i.e clean individual graphs before unioning 
def compileClaimGraph(index,claims_path,claim_IDs,clean_claims,init,end,graph_type):
	end=min(end,len(claim_IDs))
	fcg=eval(graph_type+'()')
	for claim_ID in claim_IDs[init:end]:
		filename=os.path.join(claims_path,"claim{}".format(str(claim_ID)))
		try:
			claim_g=nx.read_edgelist(filename+".edgelist",comments="@")
		except:
			continue
		claim_g=cleanClaimGraph(claim_g,clean_claims[str(claim_ID)])
		claim_g=nx.relabel_nodes(claim_g,lambda x:nodelabel_mapper(x))
		saveClaimGraph(claim_g,filename,graph_type)
		fcg.add_edges_from(claim_g.edges.data())
	return index,fcg

def compileFred(rdf_path,graph_path,graph_type,fcg_label,cpu):
	multigraph_types={'undirected':'nx.MultiGraph','directed':'nx.MultiDiGraph'}
	graph_types={'undirected':'nx.MultiGraph','directed':'nx.MultiDiGraph'}
	fcg_path=os.path.join(graph_path,"fred",fcg_label)
	#If union of tfcg and ffcg wants to be created i.e ufcg
	if fcg_label=="ufcg":
		#Assumes that tfcg and ffcg exists
		tfcg_path=os.path.join(graph_path,"fred","tfcg","tfcg.edgelist")
		ffcg_path=os.path.join(graph_path,"fred","ffcg","ffcg.edgelist")
		if os.path.exists(tfcg_path) and os.path.exists(ffcg_path):
			ufcg=eval(graph_types[graph_type]+"()")
			tfcg=nx.read_edgelist(tfcg_path,comments="@")
			ffcg=nx.read_edgelist(ffcg_path,comments="@")
			ufcg.add_edges_from(tfcg.edges.data())
			ufcg.add_edges_from(ffcg.edges.data())
			os.makedirs(fcg_path, exist_ok=True)
			saveFred(ufcg,graph_path,fcg_label)
		else:
			print("Create tfcg and ffcg before attempting to create the union: ufcg")
	else:
		claim_types={"tfcg":"true","ffcg":"false"}
		claim_type=claim_types[fcg_label]
		claims_path=os.path.join(rdf_path,"{}_claims".format(claim_type))
		claim_IDs=list(np.load(os.path.join(rdf_path,"{}_claimID.npy".format(claim_type))))
		#compiling fred only i.e. stitching together the graph using the dictionary that stores edges to remove and contract i.e. clean_claims
		with codecs.open(os.path.join(rdf_path,"{}claims_clean.json".format(claim_type)),"r","utf-8") as f: 
			clean_claims=json.loads(f.read())
			claim_IDs=list(clean_claims.keys())
		if cpu>1:
			n=int(len(claim_IDs)/cpu)+1
			pool=mp.Pool(processes=cpu)					
			results=[pool.apply_async(eval("compileClaimGraph"+str(compilefred)), args=(index,claims_path,claim_IDs,clean_claims,index*n,(index+1)*n,multigraph_types[graph_type])) for index in range(cpu)]
			output=sorted([p.get() for p in results],key=lambda x:x[0])
			fcgs=list(map(lambda x:x[1],output))
			master_fcg=eval(multigraph_types[graph_type]+'()')
			for fcg in fcgs:
				master_fcg.add_edges_from(fcg.edges.data())
		else:
			master_fcg=eval("compileClaimGraph"+str(compilefred)+"(0,claims_path,claim_IDs,clean_claims,0,"+str(len(claim_IDs))+",graph_types[graph_type])")[1]
		saveFred(master_fcg,graph_path,fcg_label)

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create fred graph')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/rdf_files/')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg','ffcg','ufcg'],help='True False or Union FactCheckGraph')
	parser.add_argument('-p','--passive',action='store_true',help='Passive or not',default=False)
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	parser.add_argument('-gt','--graphtype', metavar='Graph Type Directed/Undirected',type=str,choices=['directed','undirected'],default='undirected')
	args=parser.parse_args()
	compileFred(args.rdfpath,args.graphpath,args.graphtype,args.fcgtype,args.cpu)




