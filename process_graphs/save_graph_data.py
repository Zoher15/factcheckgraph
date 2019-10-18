import networkx as nx
import pandas as pd 
import numpy as np
import argparse
import re
import os
import codecs

def save_graph_data(graph_path,graph_class,graph_type):
	g_labels={"fred":{"true":"TFCG","false":"FFCG","union":"UFCG"},
	"co-occur":{"true":"TFCG_co","false":"FFCG_co","union":"UFCG_co"},
	"backbone":{"true":"TFCG_bb","false":"FFCG_bb","union":"UFCG_bb"},
	"kg":{"dbpedia":"DBPedia","wikidata":"Wikidata"}}
	g_label=g_labels[graph_class][graph_type]
	read_path=os.path.join(graph_path,graph_class,g_label)
	fcg=nx.read_edgelist(os.path.join(read_path,g_label+".edgelist"),data=['label'],comments="@")
	os.makedirs(os.path.join(read_path,"data"),exist_ok=True)
	write_path=os.path.join(read_path,"data",g_label)
	#Save Edges
	edges=np.asarray(list(fcg.edges()))
	np.save(write_path+"_edges.npy",edges)
	#Save Nodes
	nodes=np.asarray(list(fcg.nodes()))
	np.save(write_path+"_nodes.npy",edges)
	with codecs.open(write_path+"_edges.txt","w","utf-8") as f:
		for node in nodes:
			f.write(str(node)+"\n")
	#Save Entities
	entity_regex=re.compile(r'http:\/\/dbpedia\.org\/resource\/')
	entities=np.asarray([node for node in nodes if entity_regex.match(node)])
	np.save(write_path+"_entities.npy",entities)
	with codecs.open(write_path+"_entities.txt","w","utf-8") as f:
		for entity in entities:
			f.write(str(node)+"\n")
	#Save node2ID dictionary
	node2ID={node:i for i,node in enumerate(nodes)}
	with codecs.open(write_path+"_node2ID.json","w","utf-8") as f:
		f.write(json.dumps(node2ID,ensure_ascii=False))
	#Save Edgelist ID
	edgelistID=np.asarray([[node2ID[edge[0]],node2ID[edge[1]],1] for edge in edges])
	np.save(write_path+"_edgelistID.npy",edgelistID)

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Save data for graphs')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	parser.add_argument('-gc','--graphclass', metavar='graph class',type=str,choices=['fred','co-occur','backbone','kg'],help='Class of graph to process')
	parser.add_argument('-gt','--graphtype', metavar='graph type',type=str,choices=['tfcg','ffcg','ufcg','dbpedia','wikidata'],help='True, False, Union, DBPedia or Wikidata Graph')
	args=parser.parse_args()
	save_graph_data(args.graphpath,args.graphclass,args.graphtype)
