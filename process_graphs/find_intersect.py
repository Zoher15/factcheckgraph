import networkx as nx
import pandas as pd 
import numpy as np
import argparse
import re
import os
import codecs

def find_intersect(graph_path,fcg_class,kg_label):
	fcg_path=os.path.join(graph_path,fcg_class)
	tfcg_entities=np.load(os.path.join(fcg_path,"{}_entities.npy".format(fcg_labels[fcg_class]["true"])))
	ffcg_entities=np.load(os.path.join(fcg_path,"{}_entities.npy".format(fcg_labels[fcg_class]["false"])))
	kg_entities=np.load(os.path.join(graph_path,"kg","{}_entities.npy".format(kg_label)))
	intersect_entities=np.asarray(list(set(tfcg_entities).intersection(set(ffcg_entities))))
	intersect_entities=np.asarray(list(set(intersect_entities).intersection(set(kg_entities))))
	np.save(os.path.join(fcg_path,"intersect_entities_{}_{}.npy".format(kg_label,fcg_class)),intersect_entities)
	#Finding all possible combinations
	intersect_all_entityPairs=combinations(intersect_entities,2)
	#Converting tuples to list
	intersect_all_entityPairs=np.asarray(list(map(list,intersect_all_entityPairs)))
	np.save(os.path.join(fcg_path,"intersect_all_entityPairs_{}_{}.npy".format(kg_label,fcg_class)),intersect_all_entityPairs)
	fcg_types={"fred":["tfcg","ffcg","ufcg"],"co-occur":["tfcg_co","ffcg_co","ufcg_co"],"backbone":["tfcg_bb","ffcg_bb","ufcg_bb"]}
	fcg_labels=fcg_types[fcg_class]
	intersect_all_entityPairs_klformat=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect_all_entityPairs])
	for fcg_label in fcg_labels:
		fcg_path_data=os.path.join(fcg_path,fcg_label,"data")
		#Loading the node to ID dictionaries
		with codecs.open(os.path.join(fcg_path_data,"{}_node2ID.json".format(fcg_label)),"r","utf-8") as f:
			node2ID=json.loads(f.read())
		#Writing to file with the respective node ID
		with codecs.open(os.path.join(fcg_path_data,'intersect_all_entityPairs_{}_IDs_klformat.txt'.format(fcg_label)),"w","utf-8") as f:
			for line in intersect_all_entityPairs_klformat:
				f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(node2ID[line[1]])),str(line[2]),str(line[3]),str(int(node2ID[line[4]])),str(line[5]),str(line[6])))

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Find intersection of entities for graphs')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	parser.add_argument('-fcg','--fcgclass', metavar='fcg class',type=str,choices=['fred','co-occur','backbone'],help='Class of FactCheckGraph to process')
	parser.add_argument('-kg','--kg', metavar='knowledgegraph type',type=str,choices=['dbpedia','wikidata'],help='DBPedia or Wikidata Graph')
	args=parser.parse_args()
	find_intersect(args.graphpath,args.fcgclass,args.kg)
