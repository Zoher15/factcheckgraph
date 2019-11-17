import networkx as nx
import pandas as pd 
import numpy as np
import argparse
import re
import os
import json
import codecs
from itertools import combinations

def find_intersect(graph_path,fcg_class,kg_label):
	fcg_types={"fred":["tfcg","ffcg","ufcg"],"fred1":["tfcg1","ffcg1","ufcg1"],"fred2":["tfcg2","ffcg2","ufcg2"],"fred3":["tfcg3","ffcg3","ufcg3"],"co-occur":["tfcg_co","ffcg_co","ufcg_co"],
	"backbone_df":["tfcg_bbdf","ffcg_bbdf","ufcg_bbdf"],"backbone_dc":["tfcg_bbdc","ffcg_bbdc","ufcg_bbdc"],
	"largest_ccf":["tfcg_lgccf","ffcg_lgccf","ufcg_lgccf"],"largest_ccc":["tfcg_lgccc","ffcg_lgccc","ufcg_lgccc"],
	"old_fred":["tfcg_old","ffcg_old","ufcg_old"]}
	fcg_labels=fcg_types[fcg_class]
	fcg_path=os.path.join(graph_path,fcg_class)
	tfcg_entities=np.loadtxt(os.path.join(fcg_path,fcg_labels[0],"data","{}_entities.txt".format(fcg_labels[0])),dtype='str',encoding='utf-8')
	ffcg_entities=np.loadtxt(os.path.join(fcg_path,fcg_labels[1],"data","{}_entities.txt".format(fcg_labels[1])),dtype='str',encoding='utf-8')
	kg_entities=np.loadtxt(os.path.join(graph_path,"kg",kg_label,"data","{}_entities.txt".format(kg_label)),dtype='str',encoding='utf-8')
	intersect_entities=np.asarray(list(set(tfcg_entities).intersection(set(ffcg_entities))))
	intersect_entities=np.asarray(list(set(intersect_entities).intersection(set(kg_entities))))
	#Saving entities
	with codecs.open(os.path.join(fcg_path,"intersect_entities_{}_{}.txt".format(kg_label,fcg_class)),"w","utf-8") as f:
		for entity in intersect_entities:
			f.write(str(entity)+"\n")
	#Finding all possible combinations
	intersect_all_entityPairs=combinations(intersect_entities,2)
	#Converting tuples to list
	intersect_all_entityPairs=np.asarray(list(map(list,intersect_all_entityPairs)))
	with codecs.open(os.path.join(fcg_path,"intersect_all_entityPairs_{}_{}.txt".format(kg_label,fcg_class)),"w","utf-8") as f:
		for line in intersect_all_entityPairs:
			f.write("{} {}\n".format(str(line[0]),str(line[1])))
	intersect_all_entityPairs_klformat=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect_all_entityPairs])
	for g_label in fcg_labels+[kg_label]:
		if g_label in fcg_labels:
			g_path_data=os.path.join(fcg_path,g_label,"data")
		else:
			g_path_data=os.path.join(graph_path,"kg",g_label,"data")
		#Loading the node to ID dictionaries
		with codecs.open(os.path.join(g_path_data,"{}_node2ID.json".format(g_label)),"r","utf-8") as f:
			node2ID=json.loads(f.read())
		#Writing to file with the respective node ID
		with codecs.open(os.path.join(g_path_data,'intersect_all_entityPairs_{}_{}_{}_IDs_klformat.txt'.format(kg_label,fcg_class,g_label)),"w","utf-8") as f:
			for line in intersect_all_entityPairs_klformat:
				f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(node2ID[line[1]])),str(line[2]),str(line[3]),str(int(node2ID[line[4]])),str(line[5]),str(line[6])))

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Find intersection of entities for graphs')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-fcg','--fcgclass', metavar='fcg class',type=str,choices=['fred','fred1','fred2','fred3','co-occur','backbone_df','backbone_dc','largest_ccf','largest_ccc','old_fred'],help='Class of FactCheckGraph to process')
	parser.add_argument('-kg','--kg', metavar='knowledgegraph type',type=str,choices=['dbpedia','wikidata'],help='DBPedia or Wikidata Graph')
	args=parser.parse_args()
	find_intersect(args.graphpath,args.fcgclass,args.kg)
