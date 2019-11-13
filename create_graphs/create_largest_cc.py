import os
import re
import argparse
import networkx as nx
import codecs
import json
import numpy as np

def create_largest_cc(graph_path,fcg_class,fcg_label):
	fcg=nx.read_edgelist(os.path.join(graph_path,fcg_class,fcg_label,"{}.edgelist".format(fcg_label)),comments="@")
	if nx.is_connected(fcg):
		print("graph is is_connected")
	else:
		fcg_lg=max(nx.connected_component_subgraphs(fcg), key=len)
		#backbone graph label
		lg_label="{}_lgcc{}".format(fcg_label.split("_")[0],fcg_class[0])
		lg_path=os.path.join(graph_path,"largest_cc{}".format(fcg_class[0]),lg_label)
		os.makedirs(lg_path, exist_ok=True)
		nx.write_edgelist(fcg_lg,os.path.join(lg_path,"{}.edgelist".format(lg_label)),data=True)
		nx.write_graphml(fcg_lg,os.path.join(lg_path,"{}.graphml".format(lg_label)),prettyprint=True)
		os.makedirs(os.path.join(lg_path,"data"),exist_ok=True)
		write_path=os.path.join(lg_path,"data",lg_label)
		nodes=list(fcg_lg.nodes)
		edges=list(fcg_lg.edges)
		#Save Nodes
		with codecs.open(write_path+"_nodes.txt","w","utf-8") as f:
			for node in nodes:
				f.write(str(node)+"\n")
		#Save Entities
		entity_regex=re.compile(r'http:\/\/dbpedia\.org')
		entities=np.asarray([node for node in nodes if entity_regex.match(node)])
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
	parser = argparse.ArgumentParser(description='Create Largest Component Graph')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graphs directory',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-fcg','--fcgclass', metavar='FactCheckGraph class',type=str,help='Class of graph that already exists')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,help='True False or Union FactCheckGraph')
	args=parser.parse_args()
	create_largest_cc(args.graphpath,args.fcgclass,args.fcgtype)