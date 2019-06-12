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
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb
import sys
import os 
import json
from IPython.core.debugger import set_trace
'''
The goal of this script is the following:
1. Read uris (save_uris) and triples (save_edgelist) from the Neo4j Database
2. Create files that feed into Knowledge Linker (code for calculating shortest paths in a knowledge graph)
3. Calculate Graph Statistics
3. Plot and Calculate ROC for the given triples versus randomly generated triples
'''
#Mode can be TFCG or FFCG
mode=sys.argv[1]
#PC can be 0 local, or 1 Carbonate
pc=int(sys.argv[2])
port={"FFCG":"7687","TFCG":"11007"}
g=rdflib.Graph()
graph = Graph("bolt://127.0.0.1:"+port[mode],password="1234")
#Getting the list of degrees of the given FactCheckGraph from Neo4j
def save_degree():
	tx = graph.begin()
	degreelist=np.asarray(list(map(lambda x:x['degree'],tx.run("Match (n)-[r]-(m) with n,count(m) as degree return degree"))))
	tx.commit()
	print(tx.finished())
	np.save(os.path.join(mode,mode+"_degreelist.npy"),degreelist)
	return degreelist

#Calculate Graph statistics of interest and saves them to a file called stats.txt
def calculate_stats(degreelist,dbpedia_uris,triples_tocheck_ID):
	with open(os.path.join(mode,mode+"_stats.txt"),"w") as f:
		f.write("Number of DBpedia Uris: %s \n" % (len(dbpedia_uris)))
		f.write("Number of Triples to Check: %s \n" % (len(triples_tocheck_ID)))
		degreefreq=np.asarray([float(0) for i in range(max(degreelist)+1)])
		for degree in degreelist:
			degreefreq[degree]+=1
		degreeprob=degreefreq/sum(degreefreq)
		plt.figure()	
		plt.loglog(range(0,max(degreelist)+1),degreeprob)
		plt.xlabel('Degree')
		plt.ylabel('Probability')
		plt.title('Degree Distribution')
		plt.savefig(os.path.join(mode,mode+"_degreedist.png"))
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
		pathlen=[]
		for d in sorted(verts):
			f.write('%s %d \n' % (d, dist[d]))
			pathlen.append([d,dist[d]])
		pathlen=np.asarray(pathlen)
		np.save(os.path.join(mode,mode+"_pathlen.npy"),pathlen)
#Fetches the uris of all nodes in the give FactCheckGraph
#Do note: It creates a dictionary assigning each uri an integer ID. This is to conform to the way Knowledge Linker accepts data 
#It also finds dbpedia specific uris
def save_uris():
	matcher_node = NodeMatcher(graph)
	matcher_rel = RelationshipMatcher(graph)
	uris=list(map(lambda x:x['uri'],list(matcher_node.match())))
	np.save(os.path.join(mode,mode+"_uris.npy"),uris)
	dbpedia_uris=list(map(lambda x:x['uri'],list(matcher_node.match("dbpedia"))))
	np.save(os.path.join(mode,mode+"_dbpedia_uris.npy"),dbpedia_uris)
	with codecs.open(os.path.join(mode,mode+"_uris.txt"),"w","utf-8") as f:
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
	edgelist=[[uris_dict[triple_list[i]['n']['uri']],uris_dict[triple_list[i]['m']['uri']],1] for i in range(len(triple_list))]
	edgelist=np.asarray(edgelist)
	np.save(os.path.join(mode,mode+"_edgelist.npy"),edgelist)
	with codecs.open(os.path.join(mode,mode+'_edgelist.txt'),"w","utf-8") as f:
		for line in edgelist:
			f.write("{} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2])))
	return edgelist
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
def plot_len():
	tfcg_pathlen=np.load(os.path.join("TFCG/TFCG_pathlen.npy"))
	ffcg_pathlen=np.load(os.path.join("FFCG/FFCG_pathlen.npy"))
	x1=tfcg_pathlen[:,0]
	y1=tfcg_pathlen[:,1]
	x2=ffcg_pathlen[:,0]
	y2=ffcg_pathlen[:,1]
	plt.figure()
	plt.bar(x1,y1,label="TFCG")
	plt.bar(x2,y2,label="FFCG")
	plt.title("Distribution of Path Lengths")
	plt.xlabel('Path Length')
	plt.ylabel('Number of Times')
	plt.legend(loc="upper right")
	plt.savefig("PathLengths_times.png")
	plt.figure()
	plt.bar(x1,y1/np.sum(y1),label="TFCG")
	plt.bar(x2,y2/np.sum(y2),label="FFCG")
	plt.title("Distribution of Path Lengths")
	plt.xlabel('Path Length')
	plt.ylabel('Percentage of Times')
	plt.legend(loc="upper right")
	plt.savefig("PathLengths_percent.png")
	
def plot():
	#klinker outputs json
	title='ROC '+mode+' using Degree'
	positive=pd.read_json(os.path.join(mode,mode+"_degree_u.json"))
	positive['label']=1
	negative=pd.read_json(os.path.join(mode,mode+"_negative_degree_u.json"))
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
	plt.title(title)
	plt.savefig(title.replace(" ","_")+".png")

#It uses the output from Knowledge Linker (uses log degree as weights) and plots an ROC 
def plot_log():
	#klinker outputs json
	logpositive=pd.read_json(os.path.join(mode,mode+"_logdegree_u.json"))
	logpositive['label']=1
	lognegative=pd.read_json(os.path.join(mode,mode+"_negative_logdegree_u.json"))
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
	g = rdflib.Graph()
	g.parse('/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_graph.nt',format='nt')
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

def plot_overlap():
	#klinker outputs json
	title="Common Triples TFCG vs FFCG"
	positive=pd.read_json("Intersect_TFCG_logdegree_u.json")
	positive['label']=1
	negative=pd.read_json("Intersect_FFCG_logdegree_u.json")
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
	plt.title(title)
	plt.savefig(title.replace(" ","_")+".png")
	# intersect=pd.DataFrame(columns=['TFCG_Simil','FFCG_Simil'])
	# intersect['TFCG_Simil']=positive['simil']
	# intersect['FFCG_Simil']=negative['simil']
	intersect=positive['simil']-negative['simil']
	intersect=intersect.reset_index(drop=True)
	plt.figure()
	plt.plot(intersect)
	plt.title("Difference between Simil scores from TFCG vs FFCG")
	plt.ylabel("TFCG Simil Score - FFCG Simil Score")
	plt.xlabel("Index of common triples")
	plt.savefig("Intersect_plot.png")
	plt.figure()
	plt.hist(intersect)
	plt.title("Distribution of difference between Simil Scores in TFCG vs FFCG")
	plt.xlabel("TFCG Simil Score - FFCG Simil Score")
	plt.ylabel("Number of times")
	plt.savefig("Intersect_hist.png")
	#######For random triples
	title2="Negative Common Triples TFCG vs FFCG"
	positive2=pd.read_json("Negative_intersect_TFCG_logdegree_u.json")
	positive2['label']=1
	negative2=pd.read_json("Negative_intersect_FFCG_logdegree_u.json")
	negative2['label']=0
	positive2.filter(["simil","paths"]).sort_values(by='simil').to_csv(mode+"_paths_u_degree_+ve.csv",index=False)
	negative2.filter(["simil","paths"]).sort_values(by='simil').to_csv(mode+"_paths_u_degree_-ve.csv",index=False)
	pos_neg2=pd.concat([positive2,negative2],ignore_index=True)
	y2=list(pos_neg['label'])
	scores2=list(pos_neg['simil'])
	fpr2, tpr2, thresholds2 = metrics.roc_curve(y2, scores2, pos_label=1)
	print(metrics.auc(fpr2,tpr2))
	plt.figure()
	lw = 2
	plt.plot(fpr2, tpr2, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr2,tpr2))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.title(title2)
	plt.savefig(title2.replace(" ","_")+".png")
	# intersect=pd.DataFrame(columns=['TFCG_Simil','FFCG_Simil'])
	# intersect['TFCG_Simil']=positive['simil']
	# intersect['FFCG_Simil']=negative['simil']
	intersect2=positive2['simil']-negative2['simil']
	intersect2=intersect2.reset_index(drop=True)
	plt.figure()
	plt.plot(intersect2)
	plt.title("Difference between Random Simil scores from TFCG vs FFCG")
	plt.ylabel("TFCG Simil Score - FFCG Simil Score")
	plt.xlabel("Index of common triples")
	plt.savefig("Negative_intersect_plot.png")
	plt.figure()
	plt.hist(intersect2)
	plt.title("Distribution of difference between Random Simil Scores in TFCG vs FFCG")
	plt.xlabel("TFCG Simil Score - FFCG Simil Score")
	plt.ylabel("Number of times")
	plt.savefig("Negative_intersect_hist.png")

def overlap_triples():
	TFCG_triples_tocheck=np.load(os.path.join("TFCG","TFCG"+"_entity_triples_dbpedia.npy"))
	FFCG_triples_tocheck=np.load(os.path.join("FFCG","FFCG"+"_entity_triples_dbpedia.npy"))
	TFCG_triples_tocheck=set(map(str,list(map(list,TFCG_triples_tocheck))))
	FFCG_triples_tocheck=set(map(str,list(map(list,FFCG_triples_tocheck))))
	#intersection needs to be done on string triple and not id triples
	intersect=TFCG_triples_tocheck.intersection(FFCG_triples_tocheck)
	intersect=pd.DataFrame(map(eval,list(intersect))).drop(columns=[1])#dropping the predicate i.e. middle column 
	# intersect=list(map(np.asarray,intersect))
	intersect=intersect.values
	#Uris common from the triples from DBPedia common to both
	intersect_uris_triples=np.asarray(list(set(intersect.flatten())))
	#transforming it into triples accepted by KLinker code
	intersect=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect])
	np.save("Intersect_entity_triples_dbpedia.npy",intersect)	
	#loading FCG Uris
	TFCG_uris=np.load(os.path.join("TFCG","TFCG"+"_uris.npy"))
	TFCG_uris_dict={TFCG_uris[i]:i for i in range(len(TFCG_uris))}
	FFCG_uris=np.load(os.path.join("FFCG","FFCG"+"_uris.npy"))
	FFCG_uris_dict={FFCG_uris[i]:i for i in range(len(FFCG_uris))}
	#Uris common to both uri sets
	intersect_uris=np.asarray(list(set(TFCG_uris).intersection(set(FFCG_uris))))
	np.save("intersect_uris.npy",intersect_uris)
	np.save("intersect_uris_triples.npy",intersect_uris_triples)
	print("No. of uris common to both TFCG and FFCG:",len(intersect_uris))
	print("No. of triples common to both TFCG and FFCG:",len(intersect))
	print("No. of uris present in triples common to both TFCG and FFCG:",len(intersect_uris_triples))
	with codecs.open("TFCG/TFCG_uris_dict.json","w","utf-8") as f:
		f.write(json.dumps(TFCG_uris_dict,ensure_ascii=False))
	with codecs.open("FFCG/FFCG_uris_dict.json","w","utf-8") as f:
		f.write(json.dumps(FFCG_uris_dict,ensure_ascii=False))
	with codecs.open('Intersect_triples_TFCG_IDs.txt',"w","utf-8") as f:
		for line in intersect:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(TFCG_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(TFCG_uris_dict[line[4]])),str(line[5]),str(line[6])))
	with codecs.open('Intersect_triples_FFCG_IDs.txt',"w","utf-8") as f:
		for line in intersect:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(FFCG_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(FFCG_uris_dict[line[4]])),str(line[5]),str(line[6])))
	#############################################################################################################################
	#Negative samples
	perm=permutations(intersect_uris_triples,2)
	perms=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in perm])
	z=0
	randomlist=np.random.choice(range(len(perms)),size=len(intersect)*2,replace=False)
	negative_intersect=[]
	emptylist=[]
	for i in randomlist:
		if z<len(intersect):
			if str(list(perms[i])) in set(map(str,list(map(list,intersect)))):#eliminating random triple if it exists in the intersect set (converted individiual triples to str to make a set)
				emptylist.append(i)
			else:
				z+=1
				negative_intersect.append(perms[i])
	negative_intersect=np.asarray(negative_intersect)
	with codecs.open('Negative_intersect_triples_TFCG_IDs.txt',"w","utf-8") as f:
		for line in negative_intersect:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(TFCG_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(TFCG_uris_dict[line[4]])),str(line[5]),str(line[6])))
	with codecs.open('Negative_intersect_triples_FFCG_IDs.txt',"w","utf-8") as f:
		for line in negative_intersect:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(FFCG_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(FFCG_uris_dict[line[4]])),str(line[5]),str(line[6])))
	
##DRIVER CODE
# # Try to load stuff if files already exist
# try:
# 	uris_dict,dbpedia_uris,edgelist,degreelist=load_stuff()
# #Else process and save them
# except:
# 	uris_dict,dbpedia_uris=save_uris()
# 	edgelist=save_edgelist(uris_dict)
# 	degreelist=save_degree()
# #If 1(Carbonate) dbpedia can be parsed and triples of interest can be found
# if pc==1:
# 	try:
# 		triples_tocheck=np.load(os.path.join(mode,mode+"_entity_triples_dbpedia.npy"))
# 	except:
# 		triples_tocheck=parse_dbpedia_triples()
# 	triples_tocheck_ID=np.array([[np.nan,int(uris_dict[t[0]]),np.nan,np.nan,int(uris_dict[t[2]]),np.nan,np.nan] for t in triples_tocheck])
# 	# create_negative_samples(triples_tocheck_ID,dbpedia_uris)
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
