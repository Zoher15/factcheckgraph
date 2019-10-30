import os
import re
import argparse
import networkx as nx
import codecs
import json
import numpy as np

def create_backbone(graph_path,fcg_class,fcg_label,kg_type):
	intersect_path=os.path.join(graph_path,fcg_class,"intersect_entities_{}_{}.txt".format(kg_type,fcg_class))
	if os.path.exists(intersect_path):
		#intersect_entities file exists
		intersect_entities=np.loadtxt(intersect_path,dtype=str,encoding='utf-8')
		fcg=nx.read_edgelist(os.path.join(graph_path,fcg_class,fcg_label,"{}.edgelist".format(fcg_label)),comments="@")
		between_fcg=nx.betweenness_centrality_subset(fcg,intersect_entities,intersect_entities)
		remove_nodes_fcg=set([node for node,val in between_fcg.items() if val<=0 and node not in set(intersect_entities)])
		fcg.remove_nodes_from(remove_nodes_fcg)
		#backbone graph label
		bbg_label="{}_bb{}{}".format(fcg_label.split("_")[0],kg_type[0],fcg_class[0])
		bbg_path=os.path.join(graph_path,"backbone_{}{}".format(kg_type[0],fcg_class[0]),bbg_label)
		os.makedirs(bbg_path, exist_ok=True)
		nx.write_edgelist(fcg,os.path.join(bbg_path,"{}.edgelist".format(bbg_label)),data=True)
		nx.write_graphml(fcg,os.path.join(bbg_path,"{}.graphml".format(bbg_label)),prettyprint=True)
		os.makedirs(os.path.join(bbg_path,"data"),exist_ok=True)
		write_path=os.path.join(bbg_path,"data",bbg_label)
		nodes=list(fcg.nodes)
		edges=list(fcg.edges)
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
	else:
		print("The file for intersect_entities needs to be created")

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Create Backbone Network using betweenness centrality')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graphs directory',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	parser.add_argument('-fcg','--fcgclass', metavar='FactCheckGraph class',type=str,choices=['fred','co-occur'],help='FRED or Co-Occur')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,help='True False or Union FactCheckGraph')
	parser.add_argument('-kg','--kgtype', metavar='KnowledgeGraph type',type=str,choices=['dbpedia','wikidata'],help='DBPedia or Wikidata')
	args=parser.parse_args()
	create_backbone(args.graphpath,args.fcgclass,args.fcgtype,args.kgtype)