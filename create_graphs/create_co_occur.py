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

def create_co_occur(rdf_path,graph_path,fcg_label,skipID):
	regex_dbpedia=re.compile(r'^db:')
	regex_fredup=re.compile(r'^fu:')
	regex_entity=re.compile(r'^db:|^fu:')
	if fcg_label=="ufcg_co":
		#Assumes that TFCG_co and FFCG_co exists
		tfcg_path=os.path.join(graph_path,"co_occur","tfcg_co","tfcg_co.edgelist")
		ffcg_path=os.path.join(graph_path,"co_occur","ffcg_co","ffcg_co.edgelist")
		if os.path.exists(tfcg_path) and os.path.exists(ffcg_path):
			tfcg=nx.read_edgelist(tfcg_path,comments="@")
			ffcg=nx.read_edgelist(ffcg_path,comments="@")
			fcg_co.add_edges_from(list(tfcg.edges.data())+list(ffcg.edges.data()))
		else:
			print("Create tfcg_co and ffcg_co before attempting to create the union: ufcg_co")     
	else:
		claim_types={"tfcg_co":"true","ffcg_co":"false","covid19":"covid19","covid19topics":"covid19topics"}
		claim_type=claim_types[fcg_label]
		claim_IDs=list(np.load(os.path.join(rdf_path,"{}_claimID.npy".format(claim_type))))
		if skipID>0:
			claim_IDs.remove(skipID)
		claim_entities={}
		claim_edges={}
		# entity_regex=re.compile(r'db:')
		fcg_co=nx.MultiGraph()
		for claim_ID in claim_IDs:
			claim_entities_set=set([])
			# claim_g=rdflib.Graph()
			claim_g=nx.Graph()
			claim_nxg=nx.MultiGraph()
			# filename="claim{}.rdf".format(str(claim_ID))
			filename="claim{}_clean.edgelist".format(str(claim_ID))
			try:
				# claim_g.parse(os.path.join(rdf_path,"{}_claims".format(claim_type),filename),format='application/rdf+xml')
				claim_g=nx.read_edgelist(os.path.join(rdf_path,"{}_claims".format(claim_type),filename),comments="@")
			except:
				pass
			# for triple in claim_g:
			for edge in claim_g.edges(data=True):
				# subject,predicate,obj=list(map(str,triple))
				subject,obj,d=edge
				###FRED makes a mistake while resolving "novel coronavirus"
				if "covid19" in claim_type:
					subject=subject.replace("Middle_East_respiratory_syndrome_coronavirus","Severe_acute_respiratory_syndrome_coronavirus_2")
					obj=obj.replace("Middle_East_respiratory_syndrome_coronavirus","Severe_acute_respiratory_syndrome_coronavirus_2")
				###########################################################
				try:
					if regex_entity.match(subject):
						claim_entities_set.add(subject)
					if regex_entity.match(obj):
						claim_entities_set.add(obj)
				except KeyError:
					pass
			claim_entities[claim_ID]=list(claim_entities_set)
			claim_edges[claim_ID]=list(combinations(claim_entities[claim_ID],2))
			for edge in claim_edges[claim_ID]:
				claim_nxg.add_edge(edge[0],edge[1],claim_ID=claim_ID)
			filename=os.path.join(rdf_path,"{}_claims".format(claim_type),"claim{}_co".format(str(claim_ID)))
			nx.write_edgelist(claim_nxg,filename+".edgelist")
			fcg_co.add_edges_from(claim_nxg.edges.data())
	fcg_path=os.path.join(graph_path,"co_occur",fcg_label)
	if skipID>0:
		fcg_path=os.path.join(fcg_path,"graphs3")
		os.makedirs(fcg_path, exist_ok=True)
		nx.write_edgelist(fcg_co,os.path.join(fcg_path,"{}.edgelist".format(fcg_label+"-"+str(skipID))))
	else:
		os.makedirs(fcg_path, exist_ok=True)
		nx.write_edgelist(fcg_co,os.path.join(fcg_path,"{}.edgelist".format(fcg_label)))
		nx.write_graphml(fcg_co,os.path.join(fcg_path,"{}.graphml".format(fcg_label)),prettyprint=True)
		os.makedirs(os.path.join(fcg_path,"data"),exist_ok=True)
		write_path=os.path.join(fcg_path,"data",fcg_label)
		nodes=list(fcg_co.nodes)
		edges=list(fcg_co.edges)
		#Save Nodes
		with codecs.open(write_path+"_nodes.txt","w","utf-8") as f:
			for node in nodes:
				f.write(str(node)+"\n")
		#Save Entities
		entities=np.asarray([node for node in nodes if regex_entity.match(node)])
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


def create_co_occur2(rdf_path,graph_path,fcg_label,skipID):
	rating={'tfcg':1,'ffcg':-1,'tfcg_co':1,'ffcg_co':-1}
	regex_dbpedia=re.compile(r'^db:')
	regex_fredup=re.compile(r'^fu:')
	regex_entity=re.compile(r'^db:|^fu:')
	if fcg_label=="ufcg_co":
		#Assumes that TFCG_co and FFCG_co exists
		tfcg_path=os.path.join(graph_path,"co_occur","tfcg_co","tfcg_co.edgelist")
		ffcg_path=os.path.join(graph_path,"co_occur","ffcg_co","ffcg_co.edgelist")
		if os.path.exists(tfcg_path) and os.path.exists(ffcg_path):
			tfcg=nx.read_edgelist(tfcg_path,comments="@")
			ffcg=nx.read_edgelist(ffcg_path,comments="@")
			fcg_co=nx.MultiGraph()
			fcg_co.add_edges_from(list(tfcg.edges.data())+list(ffcg.edges.data()))
		else:
			print("Create tfcg_co and ffcg_co before attempting to create the union: ufcg_co")     
	else:
		claim_types={"tfcg_co":"true","ffcg_co":"false","covid19":"covid19","covid19topics":"covid19topics"}
		claim_type=claim_types[fcg_label]
		claim_IDs=list(np.load(os.path.join(rdf_path,"{}_claimID.npy".format(claim_type))))
		if skipID>0:
			claim_IDs.remove(skipID)
		claim_nodes={}
		claim_edges={}
		# entity_regex=re.compile(r'db:')
		fcg_co=nx.MultiGraph()
		for claim_ID in claim_IDs:
			# claim_g=rdflib.Graph()
			claim_g=nx.Graph()
			claim_nxg=nx.MultiGraph()
			# filename="claim{}.rdf".format(str(claim_ID))
			filename="claim{}_clean.edgelist".format(str(claim_ID))
			try:
				# claim_g.parse(os.path.join(rdf_path,"{}_claims".format(claim_type),filename),format='application/rdf+xml')
				claim_g=nx.read_edgelist(os.path.join(rdf_path,"{}_claims".format(claim_type),filename),comments="@")
			except:
				pass
			claim_nodes[claim_ID]=list(claim_g.nodes)
			claim_edges[claim_ID]=list(combinations(claim_nodes[claim_ID],2))
			for u,v in claim_edges[claim_ID]:
				if nx.has_path(claim_g,u,v):
					dist=nx.shortest_path_length(claim_g,source=u, target=v)
					claim_nxg.add_edge(u,v,claim_ID=claim_ID,dist=dist,rating=rating[fcg_label])
				else:
					claim_nxg.add_edge(u,v,claim_ID=claim_ID,dist=-1,rating=rating[fcg_label])
			filename=os.path.join(rdf_path,"{}_claims".format(claim_type),"claim{}_co".format(str(claim_ID)))
			nx.write_edgelist(claim_nxg,filename+".edgelist")
			fcg_co.add_edges_from(claim_nxg.edges.data())
	fcg_path=os.path.join(graph_path,"co_occur",fcg_label)
	if skipID>0:
		fcg_path=os.path.join(fcg_path,"graphs3")
		os.makedirs(fcg_path, exist_ok=True)
		nx.write_edgelist(fcg_co,os.path.join(fcg_path,"{}.edgelist".format(fcg_label+"-"+str(skipID))))
	else:
		os.makedirs(fcg_path, exist_ok=True)
		nx.write_edgelist(fcg_co,os.path.join(fcg_path,"{}.edgelist".format(fcg_label)))
		nx.write_graphml(fcg_co,os.path.join(fcg_path,"{}.graphml".format(fcg_label)),prettyprint=True)
		os.makedirs(os.path.join(fcg_path,"data"),exist_ok=True)
		write_path=os.path.join(fcg_path,"data",fcg_label)
		nodes=list(fcg_co.nodes)
		edges=list(fcg_co.edges)
		#Save Nodes
		with codecs.open(write_path+"_nodes.txt","w","utf-8") as f:
			for node in nodes:
				f.write(str(node)+"\n")
		#Save Entities
		entities=np.asarray([node for node in nodes if regex_entity.match(node)])
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

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create co-cccur graph')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/rdf_files/')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg_co','ffcg_co','ufcg_co'],help='True False or Union FactCheckGraph')
	parser.add_argument('-si','--skipID', metavar='claimID to skip',type=int,help='Claim ID to skip while creation',default=0)
	args=parser.parse_args()
	create_co_occur2(args.rdfpath,args.graphpath,args.fcgtype,args.skipID)