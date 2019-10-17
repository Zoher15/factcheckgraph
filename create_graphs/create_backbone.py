import os
import argparse
import networkx as nx
import numpy as np

def create_backbone(graph_path,graph_class,fcg_label,kg_type):
	if graph_class=="co-occur":
		fcg_label=fcg_label+"_co"
	intersect_path=os.path.join(graph_path,graph_class,"intersect_entities_{}_{}.npy".format(kg_type,graph_class))
	if os.path.exists(intersect_path):
		#intersect_entities file exists
		intersect_entities=np.load(intersect_path)
		fcg=nx.read_edgelist(os.path.join(read_path,fcg_label,"{}.edgelist".format(fcg_label)),comments="@")
		between_fcg=nx.betweenness_centrality_subset(fcg,intersect_entities,intersect_entities)
		remove_nodes_fcg=set([node for node,val in between_fcg.items() if val<=0 and node not in set(intersect_entities)])
		fcg.remove_nodes_from(remove_nodes_fcg)
		fcg_path=os.path.join(graph_path,"backbone","{}_bb".format(fcg_label))
		os.makedirs(fcg_path, exist_ok=True)
		nx.write_edgelist(fcg,os.path.join(fcg_path,"{}_bb.edgelist".format(fcg_label)),data=True)
		# nx.write_graphml(fcg,os.path.join(fcg_path,"{}.graphml".format(fcg_label)),prettyprint=True)
	else:
		print("The file for intersect_entities needs to be created")

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Create Backbone Network using betweenness centrality')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graphs directory',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	parser.add_argument('-gt','--graphclass', metavar='graph class',type=str,choices=['fred','co-occur'],help='FRED or Co-Occur')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg','ffcg','ufcg'],help='True False or Union FactCheckGraph')
	parser.add_argument('-kg','--kgtype', metavar='KnowledgeGraph type',type=str,choices=['dbpedia','wikidata'],help='DBPedia or Wikidata')
	args=parser.parse_args()
	create_backbone(args.graphpath,args.graphclass,args.fcgtype,args.kgtype)