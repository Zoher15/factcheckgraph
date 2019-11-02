import os
import argparse
import numpy as np
import networkx as nx
import rdflib
import codecs

def find_adj_pairs(graph_path,fcg_class,kg_label):
	kg=nx.read_edgelist(os.path.join(graph_path,"kg",kg_label,"{}.edgelist".format(kg_label)),comments="@")
	intersect_path=os.path.join(graph_path,fcg_class,"intersect")
	intersect_all_pairs=np.loadtxt(intersect_path+"_all_entityPairs_{}_{}.txt".format(kg_label,fcg_class),dtype=str,encoding="utf-8")
	intersect_adj=[tuple([i,pair]) for i,pair in enumerate(intersect_all_pairs) if kg.has_edge(*pair)]
	intersect_adj_ind=np.asarray(list(map(lambda x:x[0],intersect_adj)))
	intersect_adj_pairs=np.asarray(list(map(lambda x:x[1],intersect_adj)))
	intersect_nonadj_ind=np.asarray(list(set(range(len(intersect_all_pairs)))-set(intersect_adj_ind)))
	intersect_nonadj_pairs=intersect_all_pairs[intersect_nonadj_ind]
	with codecs.open(intersect_path+"_adj_entityPairs_{}_{}.txt".format(kg_label,fcg_class),"w","utf-8") as f:
		for line in intersect_adj_pairs:
			f.write("{} {}\n".format(str(line[0]),str(line[1])))
	with codecs.open(intersect_path+"_nonadj_entityPairs_{}_{}.txt".format(kg_label,fcg_class),"w","utf-8") as f:
		for line in intersect_nonadj_pairs:
			f.write("{} {}\n".format(str(line[0]),str(line[1])))
	np.save(intersect_path+"_adj_ind_{}_{}.npy".format(kg_label,fcg_class),intersect_adj_ind)
	np.save(intersect_path+"_nonadj_ind_{}_{}.npy".format(kg_label,fcg_class),intersect_nonadj_ind)

def find_adj_pairs2(graph_path,fcg_class,kg_type):
	kg=rdflib.Graph()
	kg.parse(os.path.join(graph_path,"kg",kg_label,"{}.nt".format(kg_label)))
	intersect_path=os.path.join(graph_path,fcg_class,"intersect")
	intersect_all_pairs=np.load(intersect_path+"_all_entityPairs_{}_{}.npy".format(kg_label,fcg_class))
	intersect_adj=[tuple([i,pair]) for i,pair in enumerate(intersect_all_pairs) if (pair[0],None,pair[1]) in kg or (pair[1],None,pair[0]) in kg]
	intersect_adj_ind=np.asarray(list(map(intersect_adj,lambda x:x[0])))
	intersect_adj_pairs=np.asarray(list(map(intersect_adj,lambda x:x[1])))
	intersect_nonadj_ind=np.asarray(list(set(range(len(intersect_all_pairs)))-set(intersect_adj_ind)))
	intersect_nonadj_ind=intersect_all_pairs[intersect_nonadj_ind]
	np.save(intersect_path+"_adj_entityPairs_{}_{}.npy".format(kg_label,fcg_class),intersect_adj_pairs)
	np.save(intersect_path+"_nonadj_entityPairs_{}_{}.npy".format(kg_label,fcg_class),intersect_nonadj_pairs)
	np.save(intersect_path+"_adj_ind_{}_{}.npy".format(kg_label,fcg_class),intersect_adj_ind)
	np.save(intersect_path+"_nonadj_ind_{}_{}.npy".format(kg_label,fcg_class),intersect_nonadj_ind)

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Perform closed world assumption on adjacent pairs of entities in knowledge graphs')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	parser.add_argument('-fcg','--fcgclass', metavar='fcg class',type=str,help='Class of FactCheckGraph to process')
	parser.add_argument('-kg','--kgtype', metavar='knowledgegraph type',type=str,choices=['dbpedia','wikidata'],help='DBPedia or Wikidata Graph')
	args=parser.parse_args()
	find_adj_pairs(args.graphpath,args.fcgclass,args.kgtype)
		