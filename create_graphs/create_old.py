import os
import re
import argparse
import networkx as nx
import codecs
import json
import numpy as np

def create_largest_cc(graph_path,fcg_class,fcg_label):
	write_path=os.path.join(graph_path,fcg_class,fcg_label)
	if fcg_label=="ufcg_old":
		#Assumes that tfcg and ffcg exists
		tfcg_path=os.path.join(graph_path,"old_fred","tfcg_old","tfcg_old.edgelist")
		ffcg_path=os.path.join(graph_path,"old_fred","ffcg_old","ffcg_old.edgelist")
		if os.path.exists(tfcg_path) and os.path.exists(ffcg_path):
			tfcg=nx.read_edgelist(tfcg_path,comments="@")
			ffcg=nx.read_edgelist(ffcg_path,comments="@")
			fcg=nx.compose(tfcg,ffcg)
			os.makedirs(write_path, exist_ok=True)
			nx.write_edgelist(fcg,os.path.join(write_path,"ufcg_old.edgelist"))
			nx.write_graphml(fcg,os.path.join(write_path,"ufcg_old.graphml".format(fcg_label)),prettyprint=True)
	else:
		fcg=nx.read_edgelist(os.path.join(write_path,"{}.edgelist".format(fcg_label)),comments="@")
		nx.write_graphml(fcg,os.path.join(write_path,"{}.graphml".format(fcg_label)),prettyprint=True)
	data_path=os.path.join(write_path,"data")
	os.makedirs(data_path,exist_ok=True)
	data_path=os.path.join(data_path,fcg_label)
	nodes=list(fcg.nodes)
	edges=list(fcg.edges)
	#Save Nodes
	with codecs.open(data_path+"_nodes.txt","w","utf-8") as f:
		for node in nodes:
			f.write(str(node)+"\n")
	#Save Entities
	entity_regex=re.compile(r'http:\/\/dbpedia\.org')
	entities=np.asarray([node for node in nodes if entity_regex.match(node)])
	with codecs.open(data_path+"_entities.txt","w","utf-8") as f:
		for entity in entities:
			f.write(str(entity)+"\n")
	#Save node2ID dictionary
	node2ID={node:i for i,node in enumerate(nodes)}
	with codecs.open(data_path+"_node2ID.json","w","utf-8") as f:
		f.write(json.dumps(node2ID,ensure_ascii=False))
	#Save Edgelist ID
	edgelistID=np.asarray([[node2ID[edge[0]],node2ID[edge[1]],1] for edge in edges])
	np.save(data_path+"_edgelistID.npy",edgelistID)		

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Create Largest Component Graph')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graphs directory',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	parser.add_argument('-fcg','--fcgclass', metavar='FactCheckGraph class',type=str,help='Class of graph that already exists')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,help='True False or Union FactCheckGraph')
	args=parser.parse_args()
	create_largest_cc(args.graphpath,args.fcgclass,args.fcgtype)