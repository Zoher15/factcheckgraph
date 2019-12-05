# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
import rdflib
from rdflib import BNode, Literal
from rdflib.namespace import RDF
from py2neo import Graph, NodeMatcher, RelationshipMatcher
from itertools import permutations
from sklearn import metrics
import codecs
import csv
import re
import networkx as nx 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb
import sys
import os 
'''
The goal of this script is the following:
1. Read uris (save_uris) and triples (save_edgelist) from the Neo4j Database
2. Create files that feed into Knowledge Linker (code for calculating shortest paths in a knowledge graph)
3. Calculate Graph Statistics
3. Plot and Calculate ROC for the given triples versus randomly generated triples
'''
mode=sys.argv[1]
pc=int(sys.argv[2])
name=sys.argv[3]
port={"TFCG":"7687","FFCG":"7687"}
g=rdflib.Graph()
graph = Graph("bolt://127.0.0.1:"+port[mode],password="1234")
#Getting the list of degrees of the given FactCheckGraph from Neo4j
def save_degree():
	tx = graph.begin()
	degreelist=np.asarray(list(map(lambda x:x['degree'],tx.run("Match (n)-[r]-(m) with n,count(m) as degree return degree"))))
	tx.commit()
	print(tx.finished())
	np.save(os.path.join(mode,mode+"_degreelist2.npy"),degreelist)
	return degreelist

#Calculate Graph statistics of interest and saves them to a file called stats.txt
def calculate_stats(degreelist,dbpedia_uris,triples_tocheck_ID):
	with open(os.path.join(mode,mode+"_stats.txt"),"w") as f:
		f.write("Number of DBpedia Uris: %s \n" % (len(dbpedia_uris)))
		f.write("Number of Triple to Check: %s \n" % (len(triples_tocheck_ID)))
		degreefreq=np.asarray([float(0) for i in range(max(degreelist)+1)])
		for degree in degreelist:
			degreefreq[degree]+=1
		degreeprob=degreefreq/sum(degreefreq)
		degree_square_list=np.asarray(list(map(np.square,degreelist)))
		f.write("Average Degree: %s \n" % (np.average(degreelist)))
		f.write("Average Squared Degree: %s \n" % (np.average(degree_square_list)))
		kappa=np.average(degree_square_list)/(np.square(np.average(degreelist)))
		f.write("Kappa/Heterogenity Coefficient (average of squared degree/square of average degree): %s \n" % (kappa))
		try:
			G=nx.read_weighted_edgelist(os.path.join(mode,mode+"_edgelist.txt"))
		except:
			pdb.set_trace()
		f.write("Average Clustering Coefficient: %s \n" % (nx.average_clustering(G)))
		f.write("Density: %s \n" %(nx.density(G)))
		f.write("Number of Edges: %s \n" %(G.number_of_edges()))
		f.write("Number of Nodes: %s \n" %(G.number_of_nodes()))
		#average path length calculation
		pathlengths = []
		for v in G.nodes():
			spl = dict(nx.single_source_shortest_path_length(G, v))
			for p in spl:
				pathlengths.append(spl[p])
		f.write("Average Shortest Path Length: %s \n\n" % (sum(pathlengths) / len(pathlengths)))
		dist = {}
		for p in pathlengths:
			if p in dist:
				dist[p] += 1
			else:
				dist[p] = 1
		f.write("Length #Paths \n")
		verts = dist.keys()
		for d in sorted(verts):
			f.write('%s %d \n' % (d, dist[d]))
#Fetches the uris of all nodes in the give FactCheckGraph
#Do note: It creates a dictionary assigning each uri an integer ID. This is to conform to the way Knowledge Linker accepts data 
#It also finds dbpedia specific uris
def save_uris():
	matcher_node = NodeMatcher(graph)
	matcher_rel = RelationshipMatcher(graph)
	uris=list(map(lambda x:x['uri'],list(matcher_node.match())))
	np.save(os.path.join(mode,mode+"_uris2.npy"),uris)
	dbpedia_uris=list(map(lambda x:x['uri'],list(matcher_node.match("dbpedia"))))
	np.save(os.path.join(mode,mode+"_dbpedia_uris2.npy"),dbpedia_uris)
	with codecs.open(os.path.join(mode,mode+"_uris2.txt"),"w","utf-8") as f:
		for uri in uris:
			try:
				f.write(str(uri)+"\n")
			except:
				pdb.set_trace()
	uris_dict={uris[i]:i for i in range(len(uris))}
	return uris_dict,dbpedia_uris
#If the files exist, and neo4j does not need to be accessed, simple way to load all the files of interest
def load_stuff():
	uris=np.load(os.path.join(mode,mode+"_uris.npy"))
	uris_dict={uris[i]:i for i in range(len(uris))}
	dbpedia_uris=np.load(os.path.join(mode,mode+"_dbpedia_uris.npy"))
	edgelist=np.load(os.path.join(mode,mode+"_edgelist.npy"))
	edgelist=np.matrix(edgelist)
	degreelist=np.load(os.path.join(mode,mode+"_degreelist.npy"))
	return uris_dict,dbpedia_uris,edgelist,degreelist
#Saves the graph in the form of edges as a .npy file as well as a .txt. 
#Do note: It uses an id for the uris instead of the uri itself. This is to conform with the way Knowledge Linker accepts data
def save_edgelist(uris_dict):
	tx = graph.begin()
	graph_triples=tx.run("MATCH (n)-[r]-(m) return n,r,m;")
	tx.commit()
	print(tx.finished())
	triple_list=[]
	for triple in graph_triples:
		triple_list.append(triple)
	edgelist=np.asarray([tuple(sorted([triple['n']['uri'],triple['m']['uri']])) for triple in triple_list])
	G=nx.Graph()
	np.save(os.path.join(mode,mode+"_edgelist2.npy"),edgelist)
	with codecs.open(os.path.join(mode,mode+'_edgelist2.txt'),"w","utf-8") as f:
		for line in triple_list:
			G.add_edge(str(line['n']['uri']),str(line['m']['uri']),label=str(line['r']['uri']))
			f.write("{} {} {}\n".format(str(line['n']['uri']),str(line['r']['uri']),str(line['m']['uri'])))
	nx.write_edgelist(G,name+".edgelist")
	nx.write_graphml(G,name+".graphml",prettyprint=True)
	return edgelist,G
#This function saves the dbpedia triples that we want to check as .txt
#It also generates the same number of random triples that are not part of the above set. These are referred to as negative triples
def create_negative_samples(triples_tocheck_ID,dbpedia_uris):
	with codecs.open(os.path.join(mode,mode+'_dbpedia_triples.txt'),"w","utf-8") as f:
		for line in triples_tocheck_ID:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2]),str(line[3]),str(int(line[4])),str(line[5]),str(line[6])))
	perm=permutations(dbpedia_uris,2)
	perms=np.asarray(list(map(lambda x:[np.nan,int(uris_dict[x[0]]),np.nan,np.nan,int(uris_dict[x[1]]),np.nan,np.nan],perm)))
	z=0
	randomlist=np.random.choice(range(len(perms)),size=len(triples_tocheck_ID)*2,replace=False)
	negative_triples_tocheck_ID=[]
	emptylist=[]
	for i in randomlist:
		if z<len(triples_tocheck_ID):
			if perms[i] in triples_tocheck_ID:
				emptylist.append(i)
			else:
				z+=1
				negative_triples_tocheck_ID.append(perms[i])
	negative_triples_tocheck_ID=np.asarray(negative_triples_tocheck_ID)
	with codecs.open(os.path.join(mode,mode+'_negative_dbpedia_triples.txt'),"w","utf-8") as f:
		for line in negative_triples_tocheck_ID:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2]),str(line[3]),str(int(line[4])),str(line[5]),str(line[6])))
#It uses the output from Knowledge Linker and plots an ROC 
def plot():
	#klinker outputs json
	positive=pd.read_json(mode+"_degree_u.json")
	positive['label']=1
	negative=pd.read_json(mode+"_negative_degree_u.json")
	negative['label']=0
	positive.filter(["simil","paths"]).sort_values(by='simil').to_csv(mode+"_paths_u_degree_+ve.csv",index=False)
	negative.filter(["simil","paths"]).sort_values(by='simil').to_csv(mode+"_paths_u_degree_-ve.csv",index=False)
	pos_neg=pd.concat([positive,negative],ignore_index=True)
	y=list(pos_neg['label'])
	scores=list(pos_neg['simil'])
	fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
	print(metrics.auc(fpr,tpr))
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr,tpr))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.title('ROC '+mode+' using Degree')
	plt.show()
#It uses the output from Knowledge Linker (uses log degree as weights) and plots an ROC 
def plot_log():
	#klinker outputs json
	logpositive=pd.read_json(mode+"_logdegree_u.json")
	logpositive['label']=1
	lognegative=pd.read_json(mode+"_negative_logdegree_u.json")
	lognegative['label']=0
	logpositive.filter(["simil","paths"]).sort_values(by='simil').to_csv(mode+"_paths_u_logdegree_+ve.csv",index=False)
	lognegative.filter(["simil","paths"]).sort_values(by='simil').to_csv(mode+"_paths_u_logdegree_-ve.csv",index=False)
	logpos_neg=pd.concat([logpositive,lognegative],ignore_index=True)
	logy=list(logpos_neg['label'])
	logscores=list(logpos_neg['simil'])
	logfpr, logtpr, logthresholds = metrics.roc_curve(logy, logscores, pos_label=1)
	print(metrics.auc(logfpr,logtpr))
	plt.figure()
	lw = 2
	plt.plot(logfpr, logtpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(logfpr,logtpr))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.title('ROC '+mode+' using Log Degree')
	plt.show()
#This function parses the dbpedia and tries to find the triples where the subject and object, both are in the dbpedia uris subset, i.e. they are neighbors
def parse_dbpedia_triples():
	g = Graph()
	g.parse('dbpedia_graph.nt',format='nt')
	FCG_entities=set(np.load(os.path.join(mode,mode+"_dbpedia_uris.npy")))
	triple_list=[]
	entity_hitlist=[]
	i=0
	#Looping over triples in the graph
	for triple in g:
		#splitting them into subject,predicate,object
		triple=list(map(str,triple))
		subject,predicate,obj=triple
		#Checking if subject and object is in the FCG_entities set
		if subject in FCG_entities and obj in FCG_entities:
			triple_list.append(triple)
		i+=1
	print(i)
	print(len(triple_list))
	print(len(entity_hitlist))
	np.save(os.path.join(mode,mode+"_entity_triples_dbpedia.npy"),triple_list)
	return triplelist

def stats(G):
	g_label=mode
	stats_index=["Number of Nodes","Number of Entities","Number of Edges","Number of Connected Components","Largest Component Nodes","Largest Component Entities",
	"Largest Component Edges","Average Degree","Kappa/Heterogenity Coefficient","Average Clustering Coefficient","Density","Average Shortest Path Length"]
	graph_stats=pd.DataFrame(index=stats_index,columns=[mode])
	entity_regex=re.compile(r'http:\/\/dbpedia\.org\/resource\/')
	nodes=G.nodes()
	entities=[node for node in nodes if entity_regex.match(node)]
	degree_list=[val for (node, val) in G.degree()]
	avg_degree=np.average(degree_list)
	degree_square_list=np.asarray(list(map(np.square,degree_list)))
	avg_square_degree=np.average(degree_square_list)
	graph_stats.loc["Number of Nodes",g_label]=len(list(nodes))
	graph_stats.loc["Number of Entities",g_label]=len(list(entities))
	graph_stats.loc["Number of Edges",g_label]=G.number_of_edges()
	graph_stats.loc["Number of Connected Components",g_label]=len(list(nx.connected_components(G)))
	largest_cc=max(nx.connected_component_subgraphs(G), key=len)
	largest_cc_entities=[entity for entity in largest_cc.nodes() if entity_regex.match(entity)]
	graph_stats.loc["Largest Component Nodes",g_label]=len(largest_cc.nodes())
	graph_stats.loc["Largest Component Entities",g_label]=len(largest_cc_entities)
	graph_stats.loc["Largest Component Edges",g_label]=len(largest_cc.edges())
	graph_stats.loc["Average Degree",g_label]=avg_degree
	graph_stats.loc["Average Squared Degree",g_label]=avg_square_degree
	kappa=avg_square_degree/np.square(avg_degree)
	graph_stats.loc["Kappa/Heterogenity Coefficient",g_label]=kappa
	graph_stats.loc["Average Clustering Coefficient",g_label]=nx.average_clustering(G)
	graph_stats.loc["Density",g_label]=nx.density(G)
	graph_stats.to_csv(os.path.join(mode,mode+"_stats2.csv"))
###DRIVER CODE
#Try to load stuff if files already exist
# try:
# 	uris_dict,dbpedia_uris,edgelist,degreelist=load_stuff()
# #Else process and save them
# except:
uris_dict,dbpedia_uris=save_uris()
edgelist,G=save_edgelist(uris_dict)
degreelist=save_degree()
stats(G)
# import pdb
# pdb.set_trace()
#If 1(Carbonate) dbpedia can be parsed and triples of interest can be found
# if pc==1:
# 	try:
# 		triples_tocheck=np.load(os.path.join(mode,mode+"_entity_triples_dbpedia.npy"))
# 	except:
# 		triples_tocheck=parse_dbpedia_triples()
# 	triples_tocheck_ID=np.array([[np.nan,int(uris_dict[t[0]]),np.nan,np.nan,int(uris_dict[t[2]]),np.nan,np.nan] for t in triples_tocheck])
# 	create_negative_samples(triples_tocheck_ID,dbpedia_uris)
# 	calculate_stats(degreelist,dbpedia_uris,triples_tocheck_ID)
# #Else do not try parsing dbpedia
# else:
# 	try:
# 		triples_tocheck=np.load(os.path.join(mode,mode+"_entity_triples_dbpedia.npy"))
# 		triples_tocheck_ID=np.array([[np.nan,int(uris_dict[t[0]]),np.nan,np.nan,int(uris_dict[t[2]]),np.nan,np.nan] for t in triples_tocheck])
# 	except:
# 		pass

#qsub -I -q interactive -l nodes=1:ppn=8,walltime=6:00:00,vmem=128gb
#klinker linkpred uris.txt edgelist.npy dbpedia_triples.txt tfcg_degree_u.txt -u -n 1

#klinker linkpred FFCG_uris.txt FFCG_edgelist.npy FFCG_dbpedia_triples.txt FFCG_u_degree_+ve.json -u -n 1
#klinker linkpred FFCG_uris.txt FFCG_edgelist.npy FFCG_dbpedia_triples.txt FFCG_u_logdegree_+ve.json -u -n 1 -w logdegree

#klinker linkpred TFCG_uris.txt TFCG_edgelist.npy TFCG_dbpedia_triples.txt TFCG_u_degree_+ve.json -u -n 1
#klinker linkpred TFCG_uris.txt TFCG_edgelist.npy TFCG_dbpedia_triples.txt TFCG_u_logdegree_+ve.json -u -n 1 -w logdegree
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
