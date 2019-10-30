import rdflib
import os
import argparse
import numpy as np
import networkx as nx
import codecs
import re
import json
from urllib.parse import urlparse

def create_dbpedia(graph_path):
	kg_path=os.path.join(graph_path,"kg","dbpedia")
	kg=rdflib.Graph()
	kg.parse(os.path.join(kg_path,"raw","dbpedia_2016-10.nt"),format="nt")
	kg.parse(os.path.join(kg_path,"raw","instance_types_en.ttl"),format="turtle")
	kg.parse(os.path.join(kg_path,"raw","instance_types_en2.ttl"),format="nt")
	kg.parse(os.path.join(kg_path,"raw","mappingbased_objects_en.ttl"),format="turtle")
	kg.parse(os.path.join(kg_path,"raw","mappingbased_objects_en2.ttl"),format="turtle")
	kg_nx=nx.Graph()
	removed_triples=[]
	#Removing self loops
	for triple in kg:
		t=list(map(str,triple))
		t0_urlparse=urlparse(t[0])
		t2_urlparse=urlparse(t[2])
		if t[0]==t[2] or (t0_urlparse.netloc=='' and t0_urlparse.scheme=='') or (t2_urlparse.netloc=='' and t2_urlparse.scheme==''):
			kg.remove(triple)
			removed_triples.append(t)
		else:
			kg_nx.add_edge(t[0],t[2],label=t[1])
	kg.serialize(destination=os.path.join(kg_path,'dbpedia.nt'), format='nt')
	nx.write_edgelist(kg_nx,os.path.join(kg_path,"dbpedia.edgelist"))
	#Save removed triples
	with codecs.open(write_path+"_removed_triples.txt","w","utf-8") as f:
		for triple in removed_triples:
			f.write(str(triple)+"\n")
	os.makedirs(os.path.join(kg_path,"data"),exist_ok=True)
	write_path=os.path.join(kg_path,"data","dbpedia")
	nodes=list(kg_nx.nodes)
	edges=list(kg_nx.edges)
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
	edgelistID=np.asarray([[int(node2ID[edge[0]]),int(node2ID[edge[1]]),1] for edge in edges])
	np.save(write_path+"_edgelistID.npy",edgelistID)

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Create dbpedia from turtle and nt raw formats')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	args=parser.parse_args()
	create_dbpedia(args.graphpath)
