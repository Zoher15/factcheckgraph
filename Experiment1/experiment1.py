# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
import rdflib
from rdflib import BNode, Literal
from rdflib.namespace import RDF
# from py2neo import Graph, NodeMatcher, RelationshipMatcher
from itertools import combinations
from sklearn import metrics
import codecs
import csv
import re
from scipy import stats
from decimal import Decimal
import networkx as nx 
import matplotlib
from scipy.stats import pearsonr,kendalltau,spearmanr
import tkinter
# matplotlib.use('Agg')
# font = {'family' : 'Normal',
#         'size'   :6.75}
# matplotlib.rc('font', **font)
import math
import html
import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import pdb
import sys
import os 
import json
from IPython.core.debugger import set_trace
import seaborn as sns
import fredlib as fred
from urllib.parse import urljoin
from urllib.request import pathname2url
'''
The goal of this script is the following:
1. Read uris (save_uris) and triples (save_edgelist) from the Neo4j Database
2. Create files that feed into Knowledge Linker (code for calculating shortest paths in a knowledge graph)
3. Calculate Graph Statistics
3. Plot and Calculate ROC for the given triples versus randomly generated triples
'''
#Mode can be TFCG or FFCG
# mode=sys.argv[1]
# #PC can be 0 local, or 1 Carbonate
# pc=int(sys.argv[2])
start=int(sys.argv[1])
end=int(sys.argv[2])
zo_in=int(sys.argv[3])

# port={"FFCG":"7687","TFCG":"11007"}
# g=rdflib.Graph()
# graph = Graph("bolt://127.0.0.1:"+port[mode],password="1234")
#Getting the list of degrees of the given FactCheckGraph from Neo4j
def save_degree():
	tx = graph.begin()
	degreelist=np.asarray(list(map(lambda x:x['degree'],tx.run("Match (n)-[r]-(m) with n,count(m) as degree return degree"))))
	tx.commit()
	print(tx.finished())
	np.save(os.path.join(mode,mode+"_degreelist.npy"),degreelist)
	return degreelist

#Calculate Graph statistics of interest and saves them to a file called stats.txt
def calculate_stats(mode):
	if mode=="FRED":
		modelist=['TFCG','FFCG']
	elif mode=="Co-occur":
		modelist=['TFCG_co','FFCG_co']
	degreelist={}
	pathlengths={}
	for mode in modelist:
		dbpedia_uris=np.load(os.path.join(mode,mode+"_uris.npy"))
		# triples_tocheck=np.load(os.path.join(mode,mode+"_entity_triples_dbpedia.npy"))
		G=nx.read_edgelist(os.path.join(mode,mode+".edgelist"))
		degreelist[mode]=[val for (node, val) in G.degree()]
		with open(os.path.join(mode,mode+"_stats.txt"),"w") as f:
			f.write("Number of Edges: %.2E \n" %Decimal(G.number_of_edges()))
			f.write("Number of Nodes: %.2E \n" %Decimal(G.number_of_nodes()))
			f.write("Number of Connected Components: %s \n" %(len(list(nx.connected_components(G)))))
			largest_cc = max(nx.connected_component_subgraphs(G), key=len)
			f.write("Largest Component Edges: %.2E \n" %Decimal(len(largest_cc.edges())))			
			f.write("Largest Component Nodes: %.2E \n" %Decimal(len(largest_cc.nodes())))
			f.write("Number of DBpedia Uris: %.2E \n" %Decimal(len(dbpedia_uris)))
			# f.write("Number of Triples to Check: %s \n" % (len(triples_tocheck)))
			degree_square_list=np.asarray(list(map(np.square,degreelist[mode])))
			f.write("Average Degree: %.2E \n" %Decimal(np.average(degreelist[mode])))
			f.write("Average Squared Degree: %.2E \n" %Decimal(np.average(degree_square_list)))
			kappa=np.average(degree_square_list)/(np.square(np.average(degreelist[mode])))
			f.write("Kappa/Heterogenity Coefficient: %.2E \n" %Decimal(kappa))
			f.write("Average Clustering Coefficient: %.2E \n" %Decimal(nx.average_clustering(G)))
			f.write("Density: %.2E \n" %(nx.density(G)))
			#average path length calculation
			pathlengths[mode]= []
			for v in G.nodes():
				spl = dict(nx.single_source_shortest_path_length(G, v))
				for p in spl:
					pathlengths[mode].append(spl[p])
			f.write("Average Shortest Path Length: %.2E \n\n" %Decimal(sum(pathlengths[mode]) / len(pathlengths[mode])))
			dist = {}
			for p in pathlengths[mode]:
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
			np.save(os.path.join(mode,mode+"_pathlengths.npy"),pathlengths[mode])

#Funcion to standardize all non-neutral ratings (e.g., ‘True’, ‘False’, ‘Pants on Fire’, ‘Four Pinocchios’, etc.), 
#into two possible classes (True and False), and discard all claims whose rating was neutral (e.g. ‘Mixture’)
def standardize_claims():
	data=pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/claimreviews_db2.csv",index_col=0)
	##Dropping non-str rows
	filter=list(map(lambda x:type(x)!=str,data['rating_name']))
	data.drop(data[filter].index,inplace=True)
	print(data.groupby('fact_checkerID').count())
	trueregex=re.compile(r'(?i)^true|^correct$|^mostly true$|^geppetto checkmark$')
	falseregex=re.compile(r'(?i)^false|^mostly false|^pants on fire$|^four pinocchios$|^no\ |^no:|^distorts the facts|^wrong$')
	trueind=data['rating_name'].apply(lambda x:trueregex.match(x)!=None)
	trueclaims=list(data.loc[trueind]['claimID'])
	falseind=data['rating_name'].apply(lambda x:falseregex.match(x)!=None)
	falseclaims=list(data.loc[falseind]['claimID'])
	np.save("true_claimID_list.npy",list(trueclaims))
	np.save("false_claimID_list.npy",list(falseclaims))

def create_fred_network():
	data=pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/claimreviews_db2.csv",index_col=0)
	##Dropping non-str rows
	filter=list(map(lambda x:type(x)!=str,data['rating_name']))
	data.drop(data[filter].index,inplace=True)
	print(data.groupby('fact_checkerID').count())
	trueregex=re.compile(r'(?i)^true|^correct$|^mostly true$|^geppetto checkmark$')
	falseregex=re.compile(r'(?i)^false|^mostly false|^pants on fire$|^four pinocchios$|^no\ |^no:|^distorts the facts|^wrong$')
	trueind=data['rating_name'].apply(lambda x:trueregex.match(x)!=None)
	trueclaims=data.loc[trueind]
	falseind=data['rating_name'].apply(lambda x:falseregex.match(x)!=None)
	falseclaims=data.loc[falseind]
	TFCG=nx.Graph()
	FFCG=nx.Graph()
	FCG=nx.Graph()
	TFCG_filterdata={}
	FFCG_fiterdata={}
	FCG_filterdata={}
	for index,t in trueclaims.iterrows():
		claim_text=html.unescape(t['claim_text']).replace("`","'")
		claimID=t['claimID']
		filename="/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/True Claims/Claim"+str(claimID)
		nx_graph,removed_edges,contracted_edges = fred.checkFredSentence(claim_text,"Bearer 56a28f54-7918-3fdd-9d6f-850f13bd4041",filename)
		fred.plotFredGraph(nx_graph,filename)
		TFCG=nx.union(TFCG,nx_graph)
		TFCG_filterdata[claimID]={}
		TFCG_filterdata[claimID]['removed_edges']=removed_edges
		TFCG_filterdata[claimID]['contracted_edges']=contracted_edges

	with codecs.open("TFCG/TFCG_filterdata.json","w","utf-8") as f:
		f.write(json.dumps(TFCG_filterdata,ensure_ascii=False))
	nx.write_edgelist(FFCG,os.path.join("TFCG","TFCG.edgelist"))
	set_trace()
	for t in falseclaims:
		claim_text=html.unescape(t['claim_text']).replace("`","'")
		claimID=t['claimID']
		filename="/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/False Claims/Claim"+str(claimID)
		nx_graph,removed_edges,contracted_edges = fred.checkFredSentence(claim_text,"Bearer 56a28f54-7918-3fdd-9d6f-850f13bd4041",filename)
		fred.plotFredGraph(nx_graph,filename)
		FFCG=nx.union(FFCG,nx_graph)
		FFCG_filterdata[claimID]={}
		FFCG_filterdata[claimID]['removed_edges']=removed_edges
		FFCG_filterdata[claimID]['contracted_edges']=contracted_edges
	with codecs.open("FFCG/FFCG_filterdata.json","w","utf-8") as f:
		f.write(json.dumps(FFCG_filterdata,ensure_ascii=False))
	nx.write_edgelist(FFCG,os.path.join("FFCG","FFCG.edgelist"))

#Function to create a network where nodes are entities and edges are added if they occur in the same claim
def create_cooccurrence_network():
	#Reading True and False claims list standardized by standardize_claims()
	trueclaims=np.load("true_claimID_list.npy")
	falseclaims=np.load("false_claimID_list.npy")
	trueclaim_uris,trueclaim_edges,falseclaim_uris,falseclaim_edges={}
	TFCG_co=nx.Graph()
	FFCG_co=nx.Graph()
	dbpediaregex=re.compile(r'http:\/\/dbpedia\.org\/resource\/')
	#Parsing True Claims, find dbpedia entities, adding edges and saving the graph
	for t in trueclaims:
		claim_uris=set([])
		g=rdflib.Graph()
		filename="claim"+str(t)+".rdf"
		try:
			g.parse("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/"+filename,format='application/rdf+xml')
		except:
			pass
		for triple in g:
			subject,predicate,obj=list(map(str,triple))
			try:
				if dbpediaregex.search(subject):
					claim_uris.add(subject)
				if dbpediaregex.search(obj):
					claim_uris.add(obj)
			except KeyError:
				pass
		trueclaim_uris[t]=list(claim_uris)
		trueclaim_edges[t]=list(combinations(trueclaim_uris[t],2))
		TFCG_co.add_edges_from(trueclaim_edges[t])
	nx.write_edgelist(TFCG_co,os.path.join("TFCG_co","TFCG_co.edgelist"),data=False)
	#Parsing False Claims, find dbpedia entities, adding edges and saving the graph
	for f in falseclaims:
		claim_uris=set([])
		g=rdflib.Graph()
		filename="claim"+str(f)+".rdf"
		try:
			g.parse("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/"+filename,format='application/rdf+xml')
		except:
			# continue
			pass
		for triple in g:
			subject,predicate,obj=list(map(str,triple))
			try:
				if dbpediaregex.search(subject):
					claim_uris.add(subject)
				if dbpediaregex.search(obj):
					claim_uris.add(obj)
			except KeyError:
				pass
		falseclaim_uris[f]=list(claim_uris)
		falseclaim_edges[f]=list(combinations(falseclaim_uris[f],2))
		FFCG_co.add_edges_from(falseclaim_edges[f])
	nx.write_edgelist(FFCG_co,os.path.join("FFCG_co","FFCG_co.edgelist"),data=False)

def save_uris():
	G=nx.read_weighted_edgelist(os.path.join(mode,mode+"_edgelist.txt"))
	Gc = max(nx.connected_component_subgraphs(G), key=len)
	set_trace()
	matcher_node = NodeMatcher(graph)
	matcher_rel = RelationshipMatcher(graph)
	uris=list(map(lambda x:x['uri'],list(matcher_node.match())))
	# np.save(os.path.join(mode,mode+"_uris.npy"),uris)
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
def save_edgelist(mode,uris_dict):
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
	comb=combinations(dbpedia_uris,2)
	combs=np.asarray(list(map(lambda x:[np.nan,int(uris_dict[x[0]]),np.nan,np.nan,int(uris_dict[x[1]]),np.nan,np.nan],comb)))
	z=0
	randomlist=np.random.choice(range(len(combs)),size=len(triples_tocheck_ID)*2,replace=False)
	negative_triples_tocheck_ID=[]
	emptylist=[]
	for i in randomlist:
		if z<len(triples_tocheck_ID):
			if combs[i] in triples_tocheck_ID:
				emptylist.append(i)
			else:
				z+=1
				negative_triples_tocheck_ID.append(combs[i])
	negative_triples_tocheck_ID=np.asarray(negative_triples_tocheck_ID)
	with codecs.open(os.path.join(mode,mode+'_negative_dbpedia_triples.txt'),"w","utf-8") as f:
		for line in negative_triples_tocheck_ID:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2]),str(line[3]),str(int(line[4])),str(line[5]),str(line[6])))
#It uses the output from Knowledge Linker and plots an ROC 
def plot_stats():
	lw=3
	# title="Distribution of Path Lengths"
	# tfcg_pathlengths=np.load(os.path.join("TFCG","TFCG"+"_pathlengths.npy"))
	# ffcg_pathlengths=np.load(os.path.join("FFCG","FFCG"+"_pathlengths.npy"))
	# tfcg_maxp=np.max(tfcg_pathlengths)
	# ffcg_maxp=np.max(ffcg_pathlengths)
	# plt.figure()
	# plt.hist(tfcg_pathlengths,label="TFCG",density=True,histtype='step',linewidth=lw,bins=[i+0.5 for i in range(-1,tfcg_maxp)])
	# plt.hist(ffcg_pathlengths,label="FFCG",density=True,histtype='step',linewidth=lw,bins=[i+0.5 for i in range(-1,ffcg_maxp)])
	# plt.title(title)
	# plt.xlabel('Path Length')
	# plt.ylabel('Density')
	# plt.ylim(ymin=-0.025)
	# plt.legend(loc="upper right")
	# plt.savefig(title.replace(" ","_")+".png")
	# plt.show()
	title="CCDF of Degree"
	tfcg_degreelist=np.sort(np.load(os.path.join("TFCG","TFCG"+"_degreelist.npy")))
	ffcg_degreelist=np.sort(np.load(os.path.join("FFCG","FFCG"+"_degreelist.npy")))
	y1=np.flip(np.arange(len(tfcg_degreelist)))/float(len(tfcg_degreelist)-1)
	y2=np.flip(np.arange(len(ffcg_degreelist)))/float(len(ffcg_degreelist)-1)
	plt.figure()
	# plt.plot(tfcg_degreelist,y1,label="TFCG")
	plt.plot(ffcg_degreelist,y2,label="FFCG")
	plt.legend(loc="upper right")
	# bins=np.histogram(np.log10(tfcg_degreelist + 0.5), bins='auto')
	# plt.hist(tfcg_degreelist,label="TFCG",density=True,alpha=0.5,bins=[i for i in range(int(tfcg_minp),int(tfcg_maxp)+1)])
	# plt.hist(ffcg_degreelist,label="FFCG",density=True,alpha=0.5,bins=[i for i in range(int(ffcg_minp),int(ffcg_maxp)+1)])
	plt.title(title)
	# # plt.set_xticks()
	plt.xlabel('Degree')
	plt.ylabel('Density')
	locs,labels=plt.xticks()
	# plt.xticks(locs,10**locs)
	# set_trace()
	# plt.xticks(10**plt.xticks)
	plt.show()
	# plt.savefig(title.replace(" ","_")+".png")
	# plt.show()
	# set_trace()
	# plt.figure()
	# plt.bar(maxN,y1/np.sum(y1),label="TFCG")
	# plt.bar(maxN+width,y2/np.sum(y2),label="FFCG")
	# plt.title("Distribution of Path Lengths")
	# plt.xlabel('Path Length')
	# plt.ylabel('Percentage of Times')
	# plt.legend(loc="upper right")
	# plt.savefig("PathLengths_percent.png")
	
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
	modes=['TFCG','FFCG']
	figure(figsize=(6,4))
	for i in range(2):
		# plt.subplot(1,2,i+1)
		mode=modes[i]
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
		lw = 2
		plt.plot(logfpr, logtpr,lw=lw, label=mode+' AUC:%0.2f' % metrics.auc(logfpr,logtpr))
	plt.plot([0, 1], [0, 1], color='navy', label='Baseline',lw=lw, linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
		# plt.title('ROC '+mode+' using Log Degree')
	plt.show()
	plt.savefig("TFCG_FFCG_Logdegree_ROC.png")
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
		pair=list([subject,obj])
		#Checking if subject and object is in the FCG_entities set
		if subject in FCG_entities and obj in FCG_entities:
			triple_list.append(triple)
			if str(list([subject,obj])) in set(map(str,pair_list)) or str(list([obj,subject])) in set(map(str,pair_list)):
				empty_list.append(list([subject,obj]))
			else:
				pair_list.append(pair)
		i+=1
	print(i)
	print("Triple_list:",len(triple_list))
	print("Entity_hitlist:",len(entity_hitlist))
	print("Pair_list:",len(pair_list))
	print("Empty_list:",len(empty_list))
	np.save("intersect_entity_triples_dbpedia.npy",triple_list)
	np.save("intersect_entity_pairs_dbpedia.npy",pair_list)
	np.save(os.path.join(mode,mode+"_entity_triples_dbpedia.npy"),triple_list)
	return triple_list
	return pair_list

def create_FCG():
	with codecs.open("TFCG/TFCG_uris_dict.json","r","utf-8") as f:
		TFCG_uris_dict=json.loads(f.read())
	with codecs.open("FFCG/FFCG_uris_dict.json","r","utf-8") as f:
		FFCG_uris_dict=json.loads(f.read())
	TFCG_uris_dict_ID={value:key for key,value in TFCG_uris_dict.items()}
	FFCG_uris_dict_ID={value:key for key,value in FFCG_uris_dict.items()}
	TFCG_id_edgelist=np.load(os.path.join("TFCG","TFCG"+"_edgelist.npy"))
	FFCG_id_edgelist=np.load(os.path.join("FFCG","FFCG"+"_edgelist.npy"))
	TFCG_edgelist=[tuple([TFCG_uris_dict_ID[line[0]],TFCG_uris_dict_ID[line[1]]]) for line in TFCG_id_edgelist]
	FFCG_edgelist=[tuple([FFCG_uris_dict_ID[line[0]],FFCG_uris_dict_ID[line[1]]]) for line in FFCG_id_edgelist]
	np.save(os.path.join("TFCG","TFCG"+"_edgelist_full.npy"),TFCG_edgelist)
	np.save(os.path.join("FFCG","FFCG"+"_edgelist_full.npy"),FFCG_edgelist)
	TFCG=nx.Graph()
	FFCG=nx.Graph()
	TFCG.add_edges_from(TFCG_edgelist)
	FFCG.add_edges_from(FFCG_edgelist)
	nx.write_edgelist(TFCG,os.path.join("TFCG","TFCG_full.edgelist"))
	nx.write_edgelist(FFCG,os.path.join("FFCG","FFCG_full.edgelist"))
	# with codecs.open(os.path.join("TFCG","TFCG_full.edgelist"),"w","utf-8") as f:
	# 	for line in TFCG_edgelist:
	# 		f.write("{} {}\n".format(str(line[0]),str(line[1])))
	# with codecs.open(os.path.join("FFCG","FFCG_full.edgelist"),"w","utf-8") as f:
	# 	for line in FFCG_edgelist:
	# 		f.write("{} {}\n".format(str(line[0]),str(line[1])))
	# TFCG=nx.read_edgelist(os.path.join("TFCG","TFCG"+"_edgelist_full.txt"),nodetype=str)
	# FFCG=nx.read_edgelist(os.path.join("FFCG","FFCG"+"_edgelist_full.txt"),nodetype=str)
	FCG=nx.compose(TFCG,FFCG)
	nx.write_edgelist(FCG,os.path.join("FCG","FCG_full.edgelist"))
	FCG_uris_dict={key:i for i,key in enumerate(FCG.nodes())}
	with codecs.open("FCG/FCG_uris_dict.json","w","utf-8") as f:
		f.write(json.dumps(FCG_uris_dict,ensure_ascii=False))
	# with codecs.open("FCG/FCG_uris_dict.json","r","utf-8") as f:
	# 	FCG_uris_dict=json.loads(f.read())
	with codecs.open(os.path.join("FCG",'FCG_edgelist.txt'),"w","utf-8") as f:
		for line in FCG.edges():
			f.write("{} {} {}\n".format(str(FCG_uris_dict[line[0]]),str(FCG_uris_dict[line[1]]),str(1)))


def TFCGvsFFCG():
	#We load all uris as we need to assign each a unique int ID to work with knowledge linkers
	TFCG_uris_all=np.load(os.path.join("TFCG","TFCG"+"_uris.npy"))
	FFCG_uris_all=np.load(os.path.join("FFCG","FFCG"+"_uris.npy"))
	#loading DBPedia FCG Uris as they are our focus
	TFCG_uris=np.load(os.path.join("TFCG","TFCG"+"_dbpedia_uris.npy"))
	FFCG_uris=np.load(os.path.join("FFCG","FFCG"+"_dbpedia_uris.npy"))
	#Creating dictionaries to assign unique ids to each uri for knowledge linker to process
	TFCG_uris_dict={key:i for i,key in enumerate(TFCG_uris_all)}
	FFCG_uris_dict={key:i for i,key in enumerate(FFCG_uris_all)}
	#Saving the dictionaries
	with codecs.open("TFCG/TFCG_uris_dict.json","w","utf-8") as f:
		f.write(json.dumps(TFCG_uris_dict,ensure_ascii=False))
	with codecs.open("FFCG/FFCG_uris_dict.json","w","utf-8") as f:
		f.write(json.dumps(FFCG_uris_dict,ensure_ascii=False))
	#Loading the dictionaries
	with codecs.open("TFCG/TFCG_uris_dict.json","r","utf-8") as f:
		TFCG_uris_dict=json.loads(f.read())
	with codecs.open("FFCG/FFCG_uris_dict.json","r","utf-8") as f:
		FFCG_uris_dict=json.loads(f.read())
	with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph Data/DBPedia Data/dbpedia_uris_dict.json","r","utf-8") as f:
		DBPedia_uris_dict=json.loads(f.read())
	#Performing intersection betwen TFCG, FFCG and DBPedia
	intersect_uris=np.asarray(list(set(TFCG_uris).intersection(set(FFCG_uris))))
	intersect_uris=np.asarray(list(set(intersect_uris).intersection(set(DBPedia_uris_dict.keys()))))
	#Save the intersection set
	np.save("intersect_dbpedia_uris.npy",list(intersect_uris))
	#Load the intersection set
	intersect_uris=np.load("intersect_dbpedia_uris.npy")
	#Find all possible combinations of these uris
	intersect_all_pairs=combinations(intersect_uris,2)
	intersect_all_pairs=np.asarray(list(map(list,intersect_all_pairs)))
	#These are the pairs that are "true" by our definition: They are connected in dbpedia
	intersect_true_pairs=np.load("intersect_entity_pairs_dbpedia.npy")
	#Converting to a set to eliminate duplicate pairs
	intersect_true_pairs_set=set(list(map(str,list(map(set,intersect_true_pairs)))))
	#Converting it back. Getting rid of pairs where both uris are duplicates, as well as duplicate of each pair
	intersect_true_pairs=np.asarray([i for i in list(map(list,list(map(eval,list(intersect_true_pairs_set))))) if len(i)==2])
	#Finding uris that are only part of true pairs
	intersect_true_pairs_uris=list(set(intersect_true_pairs.flatten()))
	#Choosing 2n random pairs, where n is the lenght of the total true_pairs 
	random_pairs=np.random.choice(range(len(intersect_all_pairs)),size=len(intersect_true_pairs)*2,replace=False)
	intersect_false_pairs=[]
	rejected_pairs=[]
	set_trace()
	#Rejecting pairs from random pairs that are already present in true pairs
	counter=0
	for i in random_pairs:
		if counter<len(intersect_true_pairs):
			if str(set(intersect_all_pairs[i])) in intersect_true_pairs_set or str(set(list(reversed(intersect_all_pairs[i])))) in intersect_true_pairs_set:#eliminating random triple if it exists in the intersect set (converted individiual triples to str to make a set)
				rejected_pairs.append(intersect_all_pairs[i])
			else:
				counter+=1
				intersect_false_pairs.append(intersect_all_pairs[i])
		else:
			break
	#Find all possible combinations of uris that only part of the above true pairs
	intersect_all_pairs2=combinations(intersect_true_pairs_uris,2)
	intersect_all_pairs2=np.asarray(list(map(list,intersect_all_pairs2)))
	#Choosing 2n random pairs of the intersect_true_pairs, where n is the lenght of the total true_pairs 
	random_pairs2=np.random.choice(range(len(intersect_all_pairs2)),size=len(intersect_true_pairs)*2,replace=False)
	intersect_false_pairs2=[]
	rejected_pairs2=[]
	#Rejecting pairs from random pairs that are already present in true pairs
	counter=0
	for i in random_pairs2:
		if counter<len(intersect_true_pairs):
			if str(set(intersect_all_pairs2[i])) in intersect_true_pairs_set or str(set(list(reversed(intersect_all_pairs2[i])))) in intersect_true_pairs_set:#eliminating random triple if it exists in the intersect set (converted individiual triples to str to make a set)
				rejected_pairs2.append(intersect_all_pairs2[i])
			else:
				counter+=1
				intersect_false_pairs2.append(intersect_all_pairs2[i])
		else:
			break
	intersect_true_pairs=np.asarray(intersect_true_pairs)
	intersect_false_pairs=np.asarray(intersect_false_pairs)
	intersect_false_pairs2=np.asarray(intersect_false_pairs2)
	np.save("intersect_true_pairs.npy",intersect_true_pairs)
	np.save("intersect_false_pairs.npy",intersect_false_pairs)
	np.save("intersect_false_pairs2.npy",intersect_false_pairs2)
	set_trace()
	# write_pairs_tofile(intersect_true_pairs,intersect_false_pairs,intersect_false_pairs2,TFCG_uris_dict,FFCG_uris_dict,DBPedia_uris_dict)
	write_pairs_tofile_bulk(intersect_true_pairs,intersect_false_pairs,intersect_false_pairs2,DBPedia_uris_dict)

def write_pairs_tofile(intersect_true_pairs,intersect_false_pairs,intersect_false_pairs2,TFCG_uris_dict,FFCG_uris_dict,DBPedia_uris_dict):
	# Reformatting according to the input format acccepted by Knowledge Linker
	intersect_true_pairs=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect_true_pairs])
	intersect_false_pairs=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect_false_pairs])
	intersect_false_pairs2=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect_false_pairs2])
	######################################################################Writing True Pairs
	# Writing true pairs to file using TFCG entity IDs
	with codecs.open('Intersect_true_pairs_TFCG_IDs.txt',"w","utf-8") as f:
		for line in intersect_true_pairs:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(TFCG_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(TFCG_uris_dict[line[4]])),str(line[5]),str(line[6])))
	# Writing true pairs to file using FFCG entity IDs
	with codecs.open('Intersect_true_pairs_FFCG_IDs.txt',"w","utf-8") as f:
		for line in intersect_true_pairs:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(FFCG_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(FFCG_uris_dict[line[4]])),str(line[5]),str(line[6])))
	# Writing true pairs to file using DBPedia entity IDs
	with codecs.open('Intersect_true_pairs_DBPedia_IDs.txt',"w","utf-8") as f:
		for line in intersect_true_pairs:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))
	######################################################################Writing False Pairs
	# Writing false pairs to file using TFCG entity IDs
	with codecs.open('Intersect_false_pairs_TFCG_IDs.txt',"w","utf-8") as f:
		for line in intersect_false_pairs:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(TFCG_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(TFCG_uris_dict[line[4]])),str(line[5]),str(line[6])))
	# Writing false pairs to file using FFCG entity IDs
	with codecs.open('Intersect_false_pairs_FFCG_IDs.txt',"w","utf-8") as f:
		for line in intersect_false_pairs:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(FFCG_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(FFCG_uris_dict[line[4]])),str(line[5]),str(line[6])))
	# Writing false pairs to file using DBPedia entity IDs
	with codecs.open('Intersect_false_pairs_DBPedia_IDs.txt',"w","utf-8") as f:
		for line in intersect_false_pairs:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))
	######################################################################Writing False Pairs 2
	# Writing false pairs 2 to file using TFCG entity IDs
	with codecs.open('Intersect_false_pairs2_TFCG_IDs.txt',"w","utf-8") as f:
		for line in intersect_false_pairs2:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(TFCG_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(TFCG_uris_dict[line[4]])),str(line[5]),str(line[6])))
	# Writing false pairs 2 to file using FFCG entity IDs
	with codecs.open('Intersect_false_pairs2_FFCG_IDs.txt',"w","utf-8") as f:
		for line in intersect_false_pairs2:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(FFCG_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(FFCG_uris_dict[line[4]])),str(line[5]),str(line[6])))
	# Writing false pairs 2 to file using DBPedia entity IDs
	with codecs.open('Intersect_false_pairs2_DBPedia_IDs.txt',"w","utf-8") as f:
		for line in intersect_false_pairs2:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))

def write_pairs_tofile_bulk(intersect_true_pairs,intersect_false_pairs,intersect_false_pairs2,DBPedia_uris_dict):
	# Reformatting according to the input format acccepted by Knowledge Linker
	intersect_true_pairs=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect_true_pairs])
	intersect_false_pairs=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect_false_pairs])
	intersect_false_pairs2=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect_false_pairs2])
	splits=20
	hours=1
	partition=int(len(intersect_true_pairs)/splits)
	for i in range(0,splits):
		with codecs.open(str(i+1)+'_part_intersect_true_pairs_DBPedia_IDs.txt',"w","utf-8") as f:
			for line in intersect_true_pairs[partition*i:partition*(i+1)]:
				f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))
		with codecs.open(str(i+1)+'_part_intersect_false_pairs_DBPedia_IDs.txt',"w","utf-8") as f:
			for line in intersect_false_pairs[partition*i:partition*(i+1)]:
				f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))
		with codecs.open(str(i+1)+'_part_intersect_false_pairs2_DBPedia_IDs.txt',"w","utf-8") as f:
			for line in intersect_false_pairs2[partition*i:partition*(i+1)]:
				f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))				
		with codecs.open(str(i+1)+'_job.sh',"w","utf-8") as f:
			f.write('''
#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=180gb,walltime={}:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N {}_KLinker
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_edgelist.npy {}_part_intersect_true_pairs_DBPedia_IDs.txt {}_part_intersect_true_pairs_DBPedia_IDs.json -u -n 12
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_edgelist.npy {}_part_intersect_false_pairs_DBPedia_IDs.txt {}_part_intersect_false_pairs_DBPedia_IDs.json -u -n 12
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_edgelist.npy {}_part_intersect_false_pairs2_DBPedia_IDs.txt {}_part_intersect_false_pairs2_DBPedia_IDs.json -u -n 12
				'''.format(hours,i+1,i+1,i+1,i+1,i+1,i+1,i+1))
	with codecs.open(str(splits+1)+'_part_intersect_true_pairs_DBPedia_IDs.txt',"w","utf-8") as f:
		for line in intersect_true_pairs[splits*partition:]:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))
	with codecs.open(str(splits+1)+'_part_intersect_false_pairs_DBPedia_IDs.txt',"w","utf-8") as f:
		for line in intersect_false_pairs[splits*partition:]:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))
	with codecs.open(str(splits+1)+'_part_intersect_false_pairs2_DBPedia_IDs.txt',"w","utf-8") as f:
		for line in intersect_false_pairs2[splits*partition:]:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))
	with codecs.open(str(splits+1)+'_job.sh',"w","utf-8") as f:
		f.write('''
#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=180gb,walltime={}:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N {}_KLinker
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_edgelist.npy {}_part_intersect_true_pairs_DBPedia_IDs.txt {}_part_intersect_true_pairs_DBPedia_IDs.json -u -n 12
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_edgelist.npy {}_part_intersect_false_pairs_DBPedia_IDs.txt {}_part_intersect_false_pairs_DBPedia_IDs.json -u -n 12
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_edgelist.npy {}_part_intersect_false_pairs2_DBPedia_IDs.txt {}_part_intersect_false_pairs2_DBPedia_IDs.json -u -n 12
				'''.format(hours,splits+1,splits+1,splits+1,splits+1,splits+1,splits+1,splits+1))

def write_pairs_tofile_bulk_all(intersect_all_pairs,TFCG_uris_dict,FFCG_uris_dict,DBpedia_uris_dict):
	# Reformatting according to the input format acccepted by Knowledge Linker
	intersect_all_pairs=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect_all_pairs])
	#For second pool
	splits=28
	hours=30
	partition=int(len(intersect_all_pairs)/splits)
	for i in range(splits-1,splits):	
		with codecs.open(str(i+1)+'_part_intersect_all_pairs_DBPedia_IDs.txt',"w","utf-8") as f:
			for line in intersect_all_pairs[partition*i:partition*(i+1)]:
				f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))
		with codecs.open(str(i+1)+'_job.sh',"w","utf-8") as f:
			f.write('''
#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=180gb,walltime={}:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N {}_KLinker
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_edgelist.npy {}_part_intersect_all_pairs_DBPedia_IDs.txt {}_part_intersect_all_pairs_DBPedia_IDs.json -u -n 12
				'''.format(hours,i+1,i+1,i+1))
	with codecs.open(str(splits+1)+'_part_intersect_all_pairs_DBPedia_IDs.txt',"w","utf-8") as f:
		for line in intersect_all_pairs[splits*partition:]:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))
	with codecs.open(str(splits+1)+'_job.sh',"w","utf-8") as f:
		f.write('''
#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=180gb,walltime={}:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N {}_KLinker
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_edgelist.npy {}_part_intersect_all_pairs_DBPedia_IDs.txt {}_part_intersect_all_pairs_DBPedia_IDs.json -u -n 12
			'''.format(hours,splits+1,splits+1,splits+1))

def read_pairs_fromfile_bulk():
	splits=20
	combined1=pd.DataFrame()
	combined2=pd.DataFrame()
	combined3=pd.DataFrame()
	for i in range(splits+1):
		true=pd.read_json(str(i+1)+"_part_intersect_true_pairs_DBPedia_IDs.json")
		false=pd.read_json(str(i+1)+"_part_intersect_false_pairs_DBPedia_IDs.json")
		false2=pd.read_json(str(i+1)+"_part_intersect_false_pairs2_DBPedia_IDs.json")
		print((i+1),len(true))
		print((i+1),len(false))
		print((i+1),len(false2))
		combined1=pd.concat([combined1,true],ignore_index=True)
		combined2=pd.concat([combined2,false],ignore_index=True)
		combined3=pd.concat([combined3,false2],ignore_index=True)
	print(len(combined1))
	print(len(combined2))
	print(len(combined3))
	# combined1['label']="DBPedia"
	# combined2['label']="DBPedia"
	# combined3['label']="DBPedia"
	# dbpedia_scores1=list(combined1['simil'])
	# dbpedia_scores2=list(combined2['simil'])
	# dbpedia_scores3=list(combined3['simil'])
	# np.save("Intersect_true_pairs_DBPedia_IDs.npy",dbpedia_scores1)
	# np.save("Intersect_false_pairs_DBPedia_IDs.npy",dbpedia_scores2)
	# np.save("Intersect_false_pairs2_DBPedia_IDs.npy",dbpedia_scores3)
	combined1.to_json("Intersect_true_pairs_DBPedia_IDs.json")
	combined2.to_json("Intersect_false_pairs_DBPedia_IDs.json")
	combined3.to_json("Intersect_false_pairs2_DBPedia_IDs.json")

def quantile_correlations(zo_in,start,end):
	corr_type="kendalltau"
	tfcg_scores_all=list(np.load(os.path.join("TFCG","TFCG_scores.npy")))
	ffcg_scores_all=list(np.load(os.path.join("FFCG","FFCG_scores.npy")))
	dbpedia_scores_all=list(np.load(os.path.join("DBPedia","DBPedia_scores.npy")))
	title_text=""
	# if removal:
	# 	for name in ["tfcg_scores_all","ffcg_scores_all","dbpedia_scores_all"]:
	# 		indices=[i for i, x in enumerate(eval(name)) if x == 0]
	# 		for i in sorted(indices, reverse = True):
	# 			del tfcg_scores_all[i]
	# 			del ffcg_scores_all[i]
	# 			del dbpedia_scores_all[i]
	# 		title_text="_0removed"	
	scores_all=pd.DataFrame(columns=['DBPedia','TFCG','FFCG'])
	scores_all['DBPedia']=dbpedia_scores_all
	scores_all['TFCG']=tfcg_scores_all
	scores_all['FFCG']=ffcg_scores_all
	scores_all=scores_all.sort_values(by='DBPedia',ascending=False)
	scores_all=scores_all.reset_index(drop=True)
	if end>len(scores_all):
		end=len(scores_all)
	# interval=math.floor(len(scores_all)*0.05)
	# percentiles=[0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.0]
	# dbpedia_percentiles=list(scores_all.quantile(percentiles)['DBPedia'])
	# print(len(dbpedia_percentiles))
	# print(dbpedia_percentiles)
	# dataframes_percentiles=[scores_all[scores_all.DBPedia>i] for i in dbpedia_percentiles]
	# dataframes=[scores_all.iloc[0:i+1] for i in range(len(scores_all))]
	# set_trace()
	correlations_roll=pd.DataFrame(columns=['index','type',corr_type,'p-value'])
	for i in range(start,end):
		dbpedia_tfcg=eval(corr_type)(scores_all.loc[0:i+1,'DBPedia'],scores_all.loc[0:i+1,'TFCG'])
		dbpedia_ffcg=eval(corr_type)(scores_all.loc[0:i+1,'DBPedia'],scores_all.loc[0:i+1,'FFCG'])
		tfcg_ffcg=eval(corr_type)(scores_all.loc[0:i+1,'TFCG'],scores_all.loc[0:i+1,'FFCG'])
		dbpedia_tfcg_row={'index':i+1,'type':'DBPedia - TFCG',corr_type:dbpedia_tfcg[0],'p-value':pvalue_sign(dbpedia_tfcg[1])}
		dbpedia_ffcg_row={'index':i+1,'type':'DBPedia - FFCG',corr_type:dbpedia_ffcg[0],'p-value':pvalue_sign(dbpedia_ffcg[1])}
		tfcg_ffcg_row={'index':i+1,'type':'TFCG - FFCG',corr_type:tfcg_ffcg[0],'p-value':pvalue_sign(tfcg_ffcg[1])}
		correlations_roll=correlations_roll.append(dbpedia_tfcg_row, ignore_index=True)
		correlations_roll=correlations_roll.append(dbpedia_ffcg_row, ignore_index=True)
		correlations_roll=correlations_roll.append(tfcg_ffcg_row, ignore_index=True)
	correlations_roll.to_csv("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/Correlations/{}_correlations_roll_{}_{}.csv".format(zo_in,start,end),index=False)
	# correlations_roll_dbpedia_tfcg=correlations_roll[correlations_roll['type']=="DBPedia - TFCG"].reset_index()
	# correlations_roll_dbpedia_ffcg=correlations_roll[correlations_roll['type']=="DBPedia - FFCG"].reset_index()
	# correlations_roll_tfcg_ffcg=correlations_roll[correlations_roll['type']=="TFCG - FFCG"].reset_index()

	# set_trace()
	# title="Correlations Fred"
	# plt.figure(1)
	# plt.title("FRED Network "+corr_type+title_text)
	# plt.plot([interval*i for i in range(1,21)],correlations_roll_percentile_dbpedia_tfcg[corr_type],label="DBPedia - TFCG",marker='o',linestyle='dashed')
	# for i,value in enumerate(correlations_roll_percentile_dbpedia_tfcg['p-value']):
	# 	plt.annotate(value,(percentiles[i],correlations_roll_percentile_dbpedia_tfcg[corr_type][i]))
	# plt.plot([interval*i for i in range(1,21)],correlations_roll_percentile_dbpedia_ffcg[corr_type],label="DBPedia - FFCG",marker='o',linestyle='dashed')
	# for i,value in enumerate(correlations_roll_percentile_dbpedia_ffcg['p-value']):
	# 	plt.annotate(value,(percentiles[i],correlations_roll_percentile_dbpedia_ffcg[corr_type][i]))	
	# # plt.plot(percentiles,correlations_roll_percentile[correlations_roll_percentile['type']=="TFCG - FFCG"][corr_type],label="TFCG - FFCG")
	# plt.xlabel("Decreasing Proximity Percentiles")
	# # plt.gca().invert_xaxis()
	# plt.ylabel("Correlations")
	# plt.legend(loc="upper right")
	# plt.savefig(title.replace(" ","_")+title_text+"_"+corr_type+".png")
	# plt.close()
	# plt.clf()

def plot_corr():
	title="Correlations Decreasing DBPedia Proximity"
	correlations_roll_dbpedia_tfcg=pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/Correlations/correlations_roll_dbpedia_tfcg.csv")
	correlations_roll_dbpedia_ffcg=pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/Correlations/correlations_roll_dbpedia_ffcg.csv")
	correlations_roll_tfcg_ffcg=pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/Correlations/correlations_roll_tfcg_ffcg.csv")
	fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
	fig.set_size_inches(9,4)
	ax1.set_xscale("log")
	ax1.set_title("FRED")
	ax1.plot(correlations_roll_dbpedia_tfcg['index'],correlations_roll_dbpedia_tfcg['kendalltau'],label="DBPedia - TFCG")
	ax1.plot(correlations_roll_dbpedia_ffcg['index'],correlations_roll_dbpedia_ffcg['kendalltau'],label="DBPedia - FFCG")
	ax1.set_xlabel("Number of pairs")
	ax1.set_ylabel("Correlation")
	ax1.legend(loc="upper right")
	# plt.savefig(title.replace(" ","_")+".png")
	correlations_roll_dbpedia_tfcg=pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/Correlations/correlations_co_roll_dbpedia_tfcg.csv")
	correlations_roll_dbpedia_ffcg=pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/Correlations/correlations_co_roll_dbpedia_ffcg.csv")
	correlations_roll_tfcg_ffcg=pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/Correlations/correlations_co_roll_tfcg_ffcg.csv")
	ax2.set_xscale("log")
	ax2.set_title("Co-occur")
	ax2.plot(correlations_roll_dbpedia_tfcg['index'],correlations_roll_dbpedia_tfcg['kendalltau'],label="DBPedia - TFCG")
	ax2.plot(correlations_roll_dbpedia_ffcg['index'],correlations_roll_dbpedia_ffcg['kendalltau'],label="DBPedia - FFCG")
	ax2.set_xlabel("Number of pairs")
	ax2.legend(loc="upper right")
	plt.tight_layout()
	plt.savefig(title.replace(" ","_")+".png")
	plt.close()
	plt.clf()
	# plt.show()
	plt.close()
	plt.clf()

def pvalue_sign(pvalue):
	if pvalue>0.01:
		return '!'
	else:
		return '*'

def correlations(removal):
	tfcg_scores_all=list(np.load(os.path.join("TFCG","TFCG_scores.npy")))
	ffcg_scores_all=list(np.load(os.path.join("FFCG","FFCG_scores.npy")))
	dbpedia_scores_all=list(np.load(os.path.join("DBPedia","DBPedia_scores.npy")))
	title_text=""
	if removal:
		for name in ["tfcg_scores_all","ffcg_scores_all","dbpedia_scores_all"]:
			indices=[i for i, x in enumerate(eval(name)) if x == 0]
			for i in sorted(indices, reverse = True):
				del tfcg_scores_all[i]
				del ffcg_scores_all[i]
				del dbpedia_scores_all[i]
		title_text="0 removal"
		np.save("tfcg_scores_0removed.npy",tfcg_scores_all)
		np.save("ffcg_scores_0removed.npy",ffcg_scores_all)
		np.save("dbpedia_scores_0removed.npy",dbpedia_scores_all)
	title="Proximity Distribution"
	plt.figure(1)
	# plt.hist(tfcg_scores_all,histtype='step',label='TFCG')
	# plt.hist(ffcg_scores_all,histtype='step',label='FFCG')
	# plt.hist(dbpedia_scores_all,histtype='step',label='DBPedia')
	# plt.xlabel("Proximity Scores")
	# plt.ylabel("Density")
	# plt.legend(loc="upper right")
	# plt.savefig(title.replace(" ","_")+".png")
	# plt.close()
	# plt.clf()
	sns.set_style("white")
	sns.set_style("ticks")
	ax = sns.kdeplot(pd.Series(tfcg_scores_all,name="TFCG"))
	ax = sns.kdeplot(pd.Series(ffcg_scores_all,name="FFCG"))
	ax = sns.kdeplot(pd.Series(dbpedia_scores_all,name="DBPedia"))
	plt.xlabel("Proximity Scores")
	plt.title("FRED Network "+title_text)
	plt.ylabel("Density")
	plt.savefig(title.replace(" ","_")+title_text+".png")
	plt.close()
	plt.clf()

def read_corr_fromfile_bulk():
	start=0
	n=562330
	combined=pd.DataFrame()
	for i in range(1,30):
		end=start+20000
		if end>n:
			end=n
		combined=pd.concat([combined,pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/Correlations/{}_correlations_roll_{}_{}.csv".format(i,start,end))],ignore_index=True)
		start=end
	combined=combined.sort_values(by='index').reset_index(drop=True)
	correlations_roll_dbpedia_tfcg=combined[combined['type']=="DBPedia - TFCG"].reset_index(drop=True)
	correlations_roll_dbpedia_ffcg=combined[combined['type']=="DBPedia - FFCG"].reset_index(drop=True)
	correlations_roll_tfcg_ffcg=combined[combined['type']=="TFCG - FFCG"].reset_index(drop=True)
	correlations_roll_dbpedia_tfcg.to_csv("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/Correlations/correlations_roll_dbpedia_tfcg.csv",index=False)
	correlations_roll_dbpedia_ffcg.to_csv("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/Correlations/correlations_roll_dbpedia_ffcg.csv",index=False)
	correlations_roll_tfcg_ffcg.to_csv("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/Correlations/correlations_roll_tfcg_ffcg.csv",index=False)

def write_corr_tofile_bulk():
	# Reformatting according to the input format acccepted by Knowledge Linker
	tfcg_scores_all=list(np.load(os.path.join("TFCG","TFCG_scores.npy")))
	ffcg_scores_all=list(np.load(os.path.join("FFCG","FFCG_scores.npy")))
	dbpedia_scores_all=list(np.load(os.path.join("DBPedia","DBPedia_scores.npy")))
	scores_all=pd.DataFrame(columns=['DBPedia','TFCG','FFCG'])
	scores_all['DBPedia']=dbpedia_scores_all
	scores_all['TFCG']=tfcg_scores_all
	scores_all['FFCG']=ffcg_scores_all
	n=len(scores_all)
	start=0
	for i in range(1,30):
		end=start+20000
		if end>n:
			end=n		
		with codecs.open('/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1/Correlations/'+str(i)+'_job.sh',"w","utf-8") as f:
			f.write('''
#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=5gb,walltime=2:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N {}_Corr
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1
python experiment1.py {} {} {}
				'''.format(i,start,end,i))
		start=end

def plot_TFCGvsFFCG():
	Intersect_true_TFCG=pd.read_json("Intersect_true_pairs_TFCG_IDs.json")
	Intersect_true_FFCG=pd.read_json("Intersect_true_pairs_FFCG_IDs.json")
	Intersect_true_DBPedia=pd.read_json("Intersect_true_pairs_DBPedia_IDs.json")

	Intersect_false_TFCG=pd.read_json("Intersect_false_pairs_TFCG_IDs.json")
	Intersect_false_FFCG=pd.read_json("Intersect_false_pairs_FFCG_IDs.json")
	Intersect_false_DBPedia=pd.read_json("Intersect_false_pairs_DBPedia_IDs.json")


	Intersect_false2_TFCG=pd.read_json("Intersect_false_pairs2_TFCG_IDs.json")
	Intersect_false2_FFCG=pd.read_json("Intersect_false_pairs2_FFCG_IDs.json")
	Intersect_false2_DBPedia=pd.read_json("Intersect_false_pairs2_DBPedia_IDs.json")

	Intersect_true_TFCG['label']=1
	Intersect_false_TFCG['label']=0
	Intersect_false2_TFCG['label']=0

	Intersect_true_FFCG['label']=1
	Intersect_false_FFCG['label']=0
	Intersect_false2_FFCG['label']=0

	Intersect_true_DBPedia['label']=1
	Intersect_false_DBPedia['label']=0
	Intersect_false2_DBPedia['label']=0

	true_false_TFCG=pd.concat([Intersect_true_TFCG,Intersect_false_TFCG],ignore_index=True)
	y_TFCG=list(true_false_TFCG['label'])
	scores_TFCG=list(true_false_TFCG['simil'])

	true_false2_TFCG=pd.concat([Intersect_true_TFCG,Intersect_false2_TFCG],ignore_index=True)
	y2_TFCG=list(true_false2_TFCG['label'])
	scores2_TFCG=list(true_false2_TFCG['simil'])

	true_false_FFCG=pd.concat([Intersect_true_FFCG,Intersect_false_FFCG],ignore_index=True)
	y_FFCG=list(true_false_FFCG['label'])
	scores_FFCG=list(true_false_FFCG['simil'])

	true_false2_FFCG=pd.concat([Intersect_true_FFCG,Intersect_false2_FFCG],ignore_index=True)
	y2_FFCG=list(true_false2_FFCG['label'])
	scores2_FFCG=list(true_false2_FFCG['simil'])

	true_false_DBPedia=pd.concat([Intersect_true_DBPedia,Intersect_false_DBPedia],ignore_index=True)
	y_DBPedia=list(true_false_DBPedia['label'])
	scores_DBPedia=list(true_false_DBPedia['simil'])

	true_false2_DBPedia=pd.concat([Intersect_true_DBPedia,Intersect_false2_DBPedia],ignore_index=True)
	y2_DBPedia=list(true_false2_DBPedia['label'])
	scores2_DBPedia=list(true_false2_DBPedia['simil'])

	title="True vs False Pairs"
	lw = 2
	fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
	fig.set_size_inches(9,4)
	####TFCG
	fpr, tpr, thresholds = metrics.roc_curve(y_TFCG, scores_TFCG, pos_label=1)
	print(metrics.auc(fpr,tpr))
	print("TFCG P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_true_TFCG['simil'],Intersect_false_TFCG['simil']).pvalue))
	# ax1.subplot(1,2,1)
	ax1.plot(fpr, tpr,lw=lw, label='TFCG (%0.2f) ' % metrics.auc(fpr,tpr))
	####FFCG
	fpr, tpr, thresholds = metrics.roc_curve(y_FFCG, scores_FFCG, pos_label=1)
	print(metrics.auc(fpr,tpr))
	print("FFCG P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_true_FFCG['simil'],Intersect_false_FFCG['simil']).pvalue))
	ax1.plot(fpr, tpr,lw=lw, label='FFCG (%0.2f) ' % metrics.auc(fpr,tpr))
	####DBPedia
	fpr, tpr, thresholds = metrics.roc_curve(y_DBPedia, scores_DBPedia, pos_label=1)
	print(metrics.auc(fpr,tpr))
	print("DBPedia P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_true_DBPedia['simil'],Intersect_false_DBPedia['simil']).pvalue))
	ax1.plot(fpr, tpr,lw=lw, label='DBPedia (%0.2f) ' % metrics.auc(fpr,tpr))
	ax1.plot([0, 1], [0, 1], color='navy', lw=lw,linestyle='--')
	ax1.legend(loc="lower right")
	ax1.set_title("FRED")
	ax1.set_xlabel('False Positive Rate')
	ax1.set_ylabel('True Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.savefig(title.replace(" ","_")+".png")
	# plt.close()
	# plt.clf()
	##########################False2
	# plt.subplot(1,2,2)
	####TFCG
	# fpr, tpr, thresholds = metrics.roc_curve(y2_TFCG, scores2_TFCG, pos_label=1)
	# print(metrics.auc(fpr,tpr))
	# print("TFCG P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_true_TFCG['simil'],Intersect_false_TFCG['simil']).pvalue))
	# plt.plot(fpr, tpr,lw=lw, label='TFCG (AUC = %0.2f) ' % metrics.auc(fpr,tpr))
	# ####FFCG
	# fpr, tpr, thresholds = metrics.roc_curve(y2_FFCG, scores2_FFCG, pos_label=1)
	# print(metrics.auc(fpr,tpr))
	# print("FFCG P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_true_FFCG['simil'],Intersect_false_FFCG['simil']).pvalue))
	# plt.plot(fpr, tpr,lw=lw, label='FFCG (AUC = %0.2f) ' % metrics.auc(fpr,tpr))
	# ####DBPedia
	# fpr, tpr, thresholds = metrics.roc_curve(y2_DBPedia, scores2_DBPedia, pos_label=1)
	# print(metrics.auc(fpr,tpr))
	# print("DBPedia P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_true_DBPedia['simil'],Intersect_false_DBPedia['simil']).pvalue))
	# plt.plot(fpr, tpr,lw=lw, label='DBPedia (AUC = %0.2f) ' % metrics.auc(fpr,tpr))
	# plt.plot([0, 1], [0, 1], color='navy', lw=lw, label='Baseline',linestyle='--')
	# plt.legend(loc="lower right")
	# plt.title("Co-occur")
	# plt.xlabel('False Positive Rate')
	# # plt.ylabel('True Positive Rate')
	# plt.savefig(title.replace(" ","_")+"2.png")
	# plt.close()
	# plt.clf()



	Intersect_true_TFCG=pd.read_json("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/TFCG_co/Intersect_true_pairs_TFCG_co_IDs.json")
	Intersect_true_FFCG=pd.read_json("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/FFCG_co/Intersect_true_pairs_FFCG_co_IDs.json")
	Intersect_true_DBPedia=pd.read_json("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/DBPedia/Intersect_true_pairs_DBPedia_IDs.json")

	Intersect_false_TFCG=pd.read_json("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/TFCG_co/Intersect_false_pairs_TFCG_co_IDs.json")
	Intersect_false_FFCG=pd.read_json("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/FFCG_co/Intersect_false_pairs_FFCG_co_IDs.json")
	Intersect_false_DBPedia=pd.read_json("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/DBPedia/Intersect_false_pairs_DBPedia_IDs.json")


	# Intersect_false2_TFCG=pd.read_json("/TFCG_co/Intersect_false_pairs2_TFCG_IDs.json")
	# Intersect_false2_FFCG=pd.read_json("/FFCG_co/Intersect_false_pairs2_FFCG_IDs.json")
	# Intersect_false2_DBPedia=pd.read_json("/DBPedia/Intersect_false_pairs2_DBPedia_IDs.json")

	Intersect_true_TFCG['label']=1
	Intersect_false_TFCG['label']=0
	# Intersect_false2_TFCG['label']=0

	Intersect_true_FFCG['label']=1
	Intersect_false_FFCG['label']=0
	# Intersect_false2_FFCG['label']=0

	Intersect_true_DBPedia['label']=1
	Intersect_false_DBPedia['label']=0
	# Intersect_false2_DBPedia['label']=0

	true_false_TFCG=pd.concat([Intersect_true_TFCG,Intersect_false_TFCG],ignore_index=True)
	y_TFCG=list(true_false_TFCG['label'])
	scores_TFCG=list(true_false_TFCG['simil'])

	# true_false2_TFCG=pd.concat([Intersect_true_TFCG,Intersect_false2_TFCG],ignore_index=True)
	# y2_TFCG=list(true_false2_TFCG['label'])
	# scores2_TFCG=list(true_false2_TFCG['simil'])

	true_false_FFCG=pd.concat([Intersect_true_FFCG,Intersect_false_FFCG],ignore_index=True)
	y_FFCG=list(true_false_FFCG['label'])
	scores_FFCG=list(true_false_FFCG['simil'])

	# true_false2_FFCG=pd.concat([Intersect_true_FFCG,Intersect_false2_FFCG],ignore_index=True)
	# y2_FFCG=list(true_false2_FFCG['label'])
	# scores2_FFCG=list(true_false2_FFCG['simil'])

	true_false_DBPedia=pd.concat([Intersect_true_DBPedia,Intersect_false_DBPedia],ignore_index=True)
	y_DBPedia=list(true_false_DBPedia['label'])
	scores_DBPedia=list(true_false_DBPedia['simil'])

	# true_false2_DBPedia=pd.concat([Intersect_true_DBPedia,Intersect_false2_DBPedia],ignore_index=True)
	# y2_DBPedia=list(true_false2_DBPedia['label'])
	# scores2_DBPedia=list(true_false2_DBPedia['simil'])

	title="True vs False Pairs"
	lw = 2
	# plt.figure(1)
	####TFCG
	fpr, tpr, thresholds = metrics.roc_curve(y_TFCG, scores_TFCG, pos_label=1)
	print(metrics.auc(fpr,tpr))
	print("TFCG P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_true_TFCG['simil'],Intersect_false_TFCG['simil']).pvalue))
	ax2.plot(fpr, tpr,lw=lw, label='TFCG (%0.2f) ' % metrics.auc(fpr,tpr))
	####FFCG
	fpr, tpr, thresholds = metrics.roc_curve(y_FFCG, scores_FFCG, pos_label=1)
	print(metrics.auc(fpr,tpr))
	print("FFCG P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_true_FFCG['simil'],Intersect_false_FFCG['simil']).pvalue))
	ax2.plot(fpr, tpr,lw=lw, label='FFCG (%0.2f) ' % metrics.auc(fpr,tpr))
	####DBPedia
	fpr, tpr, thresholds = metrics.roc_curve(y_DBPedia, scores_DBPedia, pos_label=1)
	print(metrics.auc(fpr,tpr))
	print("DBPedia P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_true_DBPedia['simil'],Intersect_false_DBPedia['simil']).pvalue))
	ax2.plot(fpr, tpr,lw=lw, label='DBPedia (%0.2f) ' % metrics.auc(fpr,tpr))
	ax2.plot([0, 1], [0, 1], color='navy', lw=lw,linestyle='--')
	ax2.legend(loc="lower right")
	ax2.set_title("Co-occur")
	ax2.set_xlabel('False Positive Rate')
	plt.tight_layout()
	plt.savefig(title.replace(" ","_")+".png")
	plt.close()
	plt.clf()
	# ##########################Distribution Plot
	# #Plot 3
	# plt.figure(3)
	# title="Score Difference between True and False Pairs"
	# #first overlap plot TFCG vs FFCG
	# intersect=Intersect_true_TFCG['simil']
	# intersect=intersect.reset_index(drop=True)
	# # set_trace()
	# plt.hist(Intersect_true_TFCG['simil']-Intersect_false_TFCG['simil'],density=True,histtype='step',label='TFCG')
	# plt.hist(Intersect_true_FFCG['simil']-Intersect_false_TFCG['simil'],density=True,histtype='step',label='FFCG')
	# plt.hist(Intersect_true_DBPedia['simil']-Intersect_false_DBPedia['simil'],density=True,histtype='step',label='DBPedia')	
	# plt.legend(loc="upper right")
	# plt.title(title)
	# plt.xlabel("True Pair Proximity - False Pair Proximity")
	# plt.ylabel("Density")
	# plt.savefig(title.replace(" ","_")+".png")
	# plt.close()
	# plt.clf()
	# plt.show()


# def plot_overlap():
# 	Intersect_true_TFCG=pd.read_json("Intersect_true_pairs_TFCG_IDs.json")
# 	Intersect_true_FFCG=pd.read_json("Intersect_true_pairs_FFCG_IDs.json")
# 	Intersect_true_DBPedia=pd.read_json("Intersect_true_pairs_DBPedia_IDs.json")

# 	Intersect_false_TFCG=pd.read_json("Intersect_false_pairs_TFCG_IDs.json")
# 	Intersect_false_FFCG=pd.read_json("Intersect_false_pairs_FFCG_IDs.json")
# 	Intersect_false_DBPedia=pd.read_json("Intersect_false_pairs_DBPedia_IDs.json")


# 	Intersect_false_TFCG=pd.read_json("Intersect_false_pairs2_TFCG_IDs.json")
# 	Intersect_false_FFCG=pd.read_json("Intersect_false_pairs2_FFCG_IDs.json")
# 	Intersect_false_DBPedia=pd.read_json("Intersect_false_pairs2_DBPedia_IDs.json")
# 	set_trace()
# 	# Random2_TFCG=pd.read_json("Random2_intersect_TFCG_logdegree_u.json")
# 	# Random2_FFCG=pd.read_json("Random2_intersect_FFCG_logdegree_u.json")
# 	lw = 2
# 	#klinker outputs json
# 	#########################################################################################################################
# 	#Plot 1
# 	title="Common DBPedia Triples + Random TFCG vs FFCG"
# 	plt.figure(1)
# 	#first overlap plot TFCG vs FFCG
# 	Intersect_TFCG['label']=1
# 	Intersect_FFCG['label']=0
# 	Intersect_TFCG.filter(["simil","paths"]).sort_values(by='simil').to_csv("Intersect_paths_u_degree_TFCG.csv",index=False)
# 	Intersect_FFCG.filter(["simil","paths"]).sort_values(by='simil').to_csv("Intersect_paths_u_degree_FFCG.csv",index=False)
# 	pos_neg=pd.concat([Intersect_TFCG,Intersect_FFCG],ignore_index=True)
# 	y=list(pos_neg['label'])
# 	scores=list(pos_neg['simil'])
# 	fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
# 	print(metrics.auc(fpr,tpr))
# 	print("P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_TFCG['simil'],Intersect_FFCG['simil']).pvalue))
# 	plt.plot(fpr, tpr,lw=lw, label='DBpedia TFCG vs FFCG (AUC = %0.2f)' % metrics.auc(fpr,tpr))
# 	#second overlap plot Random
# 	Random_TFCG['label']=1
# 	Random_FFCG['label']=0
# 	Random_TFCG.filter(["simil","paths"]).sort_values(by='simil').to_csv("Random_intersect_paths_u_degree_TFCG.csv",index=False)
# 	Random_FFCG.filter(["simil","paths"]).sort_values(by='simil').to_csv("Random_intersect_paths_u_degree_FFCG.csv",index=False)
# 	pos_neg=pd.concat([Random_TFCG,Random_FFCG],ignore_index=True)
# 	y=list(pos_neg['label'])
# 	scores=list(pos_neg['simil'])
# 	fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
# 	print(metrics.auc(fpr,tpr))
# 	plt.plot(fpr, tpr,lw=lw, label='Random TFCG vs FFCG (AUC = %0.2f)' % metrics.auc(fpr,tpr))
# 	#third overlap plot Random
# 	# Random2_TFCG['label']=1
# 	# Random2_FFCG['label']=0
# 	# Random2_TFCG.filter(["simil","paths"]).sort_values(by='simil').to_csv("Random_intersect_paths_u_degree_TFCG.csv",index=False)
# 	# Random2_FFCG.filter(["simil","paths"]).sort_values(by='simil').to_csv("Random_intersect_paths_u_degree_FFCG.csv",index=False)
# 	# pos_neg=pd.concat([Random2_TFCG,Random2_FFCG],ignore_index=True)
# 	# y=list(pos_neg['label'])
# 	# scores=list(pos_neg['simil'])
# 	# fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
# 	# print(metrics.auc(fpr,tpr))
# 	# plt.plot(fpr, tpr,lw=lw, label='Random2 (AUC = %0.2f)' % metrics.auc(fpr,tpr))

# 	#Basic plot stuff
# 	plt.plot([0, 1], [0, 1], color='navy', lw=lw, label='Baseline',linestyle='--')
# 	plt.xlabel('False Positive Rate')
# 	plt.ylabel('True Positive Rate')
# 	plt.legend(loc="lower right")
# 	plt.title(title)
# 	plt.savefig(title.replace(" ","_")+".png")
# 	# plt.show()
# 	plt.close()
# 	#########################################################################################################################
# 	# #Plot 2
# 	# # intersect=pd.DataFrame(columns=['TFCG_Simil','FFCG_Simil'])
# 	# # intersect['TFCG_Simil']=positive['simil']
# 	# # intersect['FFCG_Simil']=negative['simil']
# 	# title="Difference between Simil scores from TFCG vs FFCG"
# 	# plt.figure()
# 	# #first overlap plot TFCG vs FFCG
# 	# intersect=Intersect_TFCG['simil']-Intersect_FFCG['simil']
# 	# intersect=intersect.reset_index(drop=True)
# 	# plt.plot(intersect,label='TFCG vs FFCG')
# 	# #second overlap plot Random
# 	# intersect=Random_TFCG['simil']-Random_FFCG['simil']
# 	# intersect=intersect.reset_index(drop=True)
# 	# plt.plot(intersect,label='Random')
# 	# #third overlap plot Random
# 	# intersect=Random2_TFCG['simil']-Random2_FFCG['simil']
# 	# intersect=intersect.reset_index(drop=True)
# 	# plt.plot(intersect,label='Random2')
# 	# #Basic plot stuff
# 	# plt.legend(loc="upper right")
# 	# plt.title(title)
# 	# plt.ylabel("TFCG Simil Score - FFCG Simil Score")
# 	# plt.xlabel("Index of common triples")
# 	# plt.show()
# 	# plt.savefig(title.replace(" ","_")+".png")
# 	#########################################################################################################################
# 	#Plot 3
# 	title="Distribution of difference between Simil Scores in TFCG vs FFCG"
# 	plt.figure(2)
# 	#first overlap plot TFCG vs FFCG
# 	intersect=Intersect_TFCG['simil']-Intersect_FFCG['simil']
# 	intersect=intersect.reset_index(drop=True)
# 	plt.hist(intersect,density=True,histtype='step',label='DBPedia TFCG vs FFCG')
# 	#second overlap plot Random
# 	intersect=Random_TFCG['simil']-Random_FFCG['simil']
# 	intersect=intersect.reset_index(drop=True)
# 	plt.hist(intersect,density=True,histtype='step',label='Random TFCG vs FFCG')
# 	#third overlap plot Random
# 	# intersect=Random2_TFCG['simil']-Random2_FFCG['simil']
# 	# intersect=intersect.reset_index(drop=True)
# 	# plt.hist(intersect,density=True,histtype='step',label='Random2')
# 	#Basic plot stuff
# 	plt.legend(loc="upper right")
# 	plt.title(title)
# 	plt.xlabel("TFCG Simil Score - FFCG Simil Score")
# 	plt.ylabel("Density")
# 	plt.savefig(title.replace(" ","_")+".png")
# 	# plt.show()
# 	plt.close()
# 	#########################################################################################################################
# 	#Plot 4
# 	title="TFCG+FFCG DBpedia-Neighbor-Triples vs Random Triples"
# 	plt.figure(3)
# 	#First overlap TFCG
# 	Intersect_TFCG['label']=1
# 	Random_TFCG['label']=0
# 	pos_neg=pd.concat([Intersect_TFCG,Random_TFCG],ignore_index=True)
# 	y=list(pos_neg['label'])
# 	scores=list(pos_neg['simil'])
# 	fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
# 	print(metrics.auc(fpr,tpr))
# 	plt.plot(fpr, tpr,lw=lw, label='TFCG DBPedia vs Random (AUC = %0.2f)' % metrics.auc(fpr,tpr))
# 	#second overlap FFCG
# 	Intersect_FFCG['label']=1
# 	Random_FFCG['label']=0
# 	pos_neg=pd.concat([Intersect_FFCG,Random_FFCG],ignore_index=True)
# 	y=list(pos_neg['label'])
# 	scores=list(pos_neg['simil'])
# 	fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
# 	print(metrics.auc(fpr,tpr))
# 	plt.plot(fpr, tpr,lw=lw, label='FFCG DBPedia vs Random (AUC = %0.2f)' % metrics.auc(fpr,tpr))
# 	#Basic plot stuff
# 	plt.plot([0, 1], [0, 1], color='navy', lw=lw, label='Baseline',linestyle='--')
# 	plt.xlabel('False Positive Rate')
# 	plt.ylabel('True Positive Rate')
# 	plt.legend(loc="lower right")
# 	plt.title(title)
# 	plt.savefig(title.replace(" ","_")+".png")
# 	# plt.show()
# 	plt.close()
# #DRIVER CODE
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
# calculate_stats()
# quantile_correlations(zo_in,start,end)