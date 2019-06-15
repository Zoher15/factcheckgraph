# -*- coding: utf-8 -*-
import rdflib
import os
import numpy as np
import json
import pandas as pd
import re
import pdb
import codecs
import networkx as nx 
from itertools import combinations,chain
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import metrics
from IPython.core.debugger import set_trace
'''
The goal of the script is the following:
1. Parse DBpedia and create a dictionary of uris
2. Parse DBPedia and convert to an edgelist
3. Parse every claim and extract the dbpedia entities in each of them
'''
#Function to parse dbpedia, get uris create an ID dictionary and save it in the form of edgelist
#This format is to enable use of Knowledge Linker. Hence a uris.txt file is created for index 
def parse_dbpedia():
	g = rdflib.Graph()
	g.parse('/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_graph.nt',format='nt')
	uris=set([])
	uris_dict={}
	edgelist=set([])
	i=0
	#Looping over triples in the graph
	for triple in g:
		#splitting them into subject,predicate,object
		triple=list(map(str,triple))
		subject,predicate,obj=triple
		#if subject and object have already been seen
		if subject in uris and obj in uris:
			subjid=uris_dict[subject]
			objid=uris_dict[obj]
			if tuple([subjid,objid,1]) not in edgelist and tuple([objid,subjid,1]) not in edgelist:
				edgelist.add(tuple([subjid,objid,1]))
		#if only subject has been seen
		elif subject in uris:
			subjid=uris_dict[subject]
			objid=len(uris)
			uris.add(obj)
			uris_dict[obj]=objid
			edgelist.add(tuple([subjid,objid,1]))
		#if only object has been seen before
		elif obj in uris:
			objid=uris_dict[obj]
			subjid=len(uris)
			uris.add(subject)
			uris_dict[subject]=subjid
			edgelist.add(tuple([subjid,objid,1]))
		#if neither have been seen before
		else:
			subjid=len(uris)
			uris.add(subject)
			uris_dict[subject]=subjid
			objid=len(uris)
			uris.add(obj)
			uris_dict[obj]=objid
			edgelist.add(tuple([subjid,objid,1]))
		i+=1
	print(i)
	print(len(edgelist))
	print(len(uris))
	uris=list(uris)
	edgelist=list(edgelist)
	edgelist=np.asarray([list(i) for i in edgelist])
	np.save("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_uris.npy",uris)
	np.save("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_edgelist.npy",edgelist)
	with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_uris_dict.json","w","utf-8") as f:
		f.write(json.dumps(uris_dict,ensure_ascii=False))
	with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_uris.txt","w","utf-8") as f:
		for uri in uris_dict.keys():
			try:
				f.write(str(uri)+"\n")
			except:
				pdb.set_trace()
	with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_edgelist.txt","w","utf-8") as f:
		for line in edgelist:
			f.write("{} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2])))
	return uris,uris_dict,edgelist

#Go through True and False Claims to return the entities present per claim
def parse_claims(uris_dict):
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
	trueclaim_uris={}
	falseclaim_uris={}
	dbpediaregex=re.compile(r'http:\/\/dbpedia\.org\/resource\/')
	for t in trueclaims:
		claim_uris=set([])
		g=rdflib.Graph()
		filename="claim"+str(t)+".rdf"
		try:
			g.parse("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/"+filename,format='application/rdf+xml')
		except:
			# continue
			pass
		for triple in g:
			subject,predicate,obj=list(map(str,triple))
			try:
				if dbpediaregex.search(subject):
					claim_uris.add(uris_dict[subject])
				if dbpediaregex.search(obj):
					claim_uris.add(uris_dict[obj])
			except KeyError:
				continue
		trueclaim_uris[t]=list(claim_uris)
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
					claim_uris.add(uris_dict[subject])
				if dbpediaregex.search(obj):
					claim_uris.add(uris_dict[obj])
			except KeyError:
				continue
		falseclaim_uris[f]=list(claim_uris)
	with codecs.open("trueclaim_uris.json","w","utf-8") as f:
		f.write(json.dumps(trueclaim_uris,ensure_ascii=False))
	with codecs.open("falseclaim_uris.json","w","utf-8") as f:
		f.write(json.dumps(falseclaim_uris,ensure_ascii=False))

def calculate_DBPedia_stats():
	with open("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/DBPedia_stats2.txt","w") as f:
		try:
			G=nx.read_weighted_edgelist("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_edgelist.txt")
		except:
			pdb.set_trace()
		degreelist=list(G.degree())
		degreelist=list(map(lambda x:x[1],degreelist))
		f.write("Number of DBpedia Uris: %s \n" % (len(G)))
		degreefreq=np.asarray([float(0) for i in range(max(degreelist)+1)])
		for degree in degreelist:
			degreefreq[degree]+=1
		degreeprob=degreefreq/sum(degreefreq)
		plt.figure()	
		plt.loglog(range(0,max(degreelist)+1),degreeprob)
		plt.xlabel('Degree')
		plt.ylabel('Probability')
		plt.title('Degree Distribution')
		plt.savefig("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_degreedist.png")
		degree_square_list=np.asarray(list(map(np.square,degreelist)))
		f.write("Average Degree: %s \n" % (np.average(degreelist)))
		f.write("Average Squared Degree: %s \n" % (np.average(degree_square_list)))
		kappa=np.average(degree_square_list)/(np.square(np.average(degreelist)))
		f.write("Kappa/Heterogenity Coefficient (average of squared degree/square of average degree): %s \n" % (kappa))
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
		x=pathlen[:,0]
		y=pathlen[:,1]
		plt.figure()	
		plt.bar(x,y)
		plt.xlabel('Path Length')
		plt.ylabel('Number of Times')
		plt.title("DBPedia Distribution of Path Lengths")
		plt.savefig("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_pathlen.png")
		np.save("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_pathlen.npy",pathlen)
#This function loads already saved files
def load_stuff():
	# uris=np.load("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_uris.npy")
	# with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_uris_dict.json","r","utf-8") as f:
	# 	uris_dict=json.loads(f.read())
	# with codecs.open("trueclaim_uris.json","r","utf-8") as f:
	# 	trueclaim_uris=json.loads(f.read())
	# with codecs.open("falseclaim_uris.json","r","utf-8") as f:
	# 	falseclaim_uris=json.loads(f.read())
	# with codecs.open("falseclaim_map.json","r","utf-8") as f:
	# 	falseclaim_map=json.loads(f.read())	
	# with codecs.open("trueclaim_map.json","r","utf-8") as f:
	# 	trueclaim_map=json.loads(f.read())	
	# edgelist=np.load("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_edgelist.npy")
	trueclaim_inputlist=np.load("trueclaim_inputlist.npy")
	falseclaim_inputlist=np.load("falseclaim_inputlist.npy")
	# return uris,uris_dict,trueclaim_uris,falseclaim_uris,trueclaim_map,falseclaim_map,trueclaim_inputlist,falseclaim_inputlist#,edgelist
	return trueclaim_inputlist,falseclaim_inputlist
#This function uses the entities per claim, and creates all possible combination of 2
#It also creates a list of triples and creates a map of which claim corresponds to which index of triple
def create_input_file(trueclaim_uris,falseclaim_uris):
	trueclaim_map={}
	falseclaim_map={}
	trueclaim_inputlist=[]
	falseclaim_inputlist=[]
	i=0
	with codecs.open('trueclaim_triples.txt',"w","utf-8") as f:
		for key in trueclaim_uris.keys():
			comb=list(combinations(trueclaim_uris[key],2))
			#new_i is the number of combinations produce. Want to maintain the list of index of triples in the list, that map to a particular claim
			new_i=i+len(comb)
			trueclaim_map[key]=tuple([i,new_i])
			i=new_i
			for c in comb:
				line=[np.nan,int(c[0]),np.nan,np.nan,int(c[1]),np.nan,np.nan]
				trueclaim_inputlist.append(line)
				f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2]),str(line[3]),str(int(line[4])),str(line[5]),str(line[6])))
	np.save("trueclaim_inputlist.npy",trueclaim_inputlist)
	with codecs.open("trueclaim_map.json","w","utf-8") as f:
		f.write(json.dumps(trueclaim_map,ensure_ascii=False))
	i=0
	with codecs.open('falseclaim_triples.txt',"w","utf-8") as f:
		for key in falseclaim_uris.keys():
			comb=list(combinations(falseclaim_uris[key],2))
			#new_i is the number of combinations produce. Want to maintain the list of index of triples in the list, that map to a particular claim
			new_i=i+len(comb)
			falseclaim_map[key]=tuple([i,new_i])
			i=new_i
			for c in comb:
				line=[np.nan,int(c[0]),np.nan,np.nan,int(c[1]),np.nan,np.nan]
				falseclaim_inputlist.append(line)
				f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2]),str(line[3]),str(int(line[4])),str(line[5]),str(line[6])))
	np.save("falseclaim_inputlist.npy",falseclaim_inputlist)
	with codecs.open("falseclaim_map.json","w","utf-8") as f:
		f.write(json.dumps(falseclaim_map,ensure_ascii=False))
#This function simply calculates and writes stats for TFCG and FFCG
def calculate_stats(trueclaim_uris,falseclaim_uris,trueclaim_inputlist,falseclaim_inputlist):
	with open("trueclaims_stats.txt","w") as f:
		trueclaim_list=[z for z in trueclaim_uris.values() if len(z)>0]
		f.write("Number of True Claims containing DBPedia Uris: %s \n" % (len(trueclaim_list)))
		trueclaim_totaluris=list(chain(*trueclaim_list))
		f.write("Total number of DBPedia Uris across all True claims: %s \n" % (len(trueclaim_totaluris)))
		f.write("Total number of Triples across all True claims: %s \n" % (len(trueclaim_inputlist)))
	with open("falseclaims_stats.txt","w") as f:
		falseclaim_list=[z for z in falseclaim_uris.values() if len(z)>0]
		f.write("Number of False Claims containing DBPedia Uris: %s \n" % (len(falseclaim_list)))
		falseclaim_totaluris=list(chain(*falseclaim_list))
		f.write("Total number of DBPedia Uris across all False claims: %s \n" % (len(falseclaim_totaluris)))
		f.write("Total number of Triples across all False claims: %s \n" % (len(falseclaim_inputlist)))

#It uses the output from Knowledge Linker and plots an ROC 
def plot(trueclaim_map,falseclaim_map):
	#klinker outputs json
	positive=pd.read_json("trueclaim_degree_u.json")
	positive['label']=1
	negative=pd.read_json("falseclaim_degree_u.json")
	negative['label']=0
	positive.filter(["simil","paths"]).sort_values(by='simil').to_csv("TrueClaims_paths.csv",index=False)
	negative.filter(["simil","paths"]).sort_values(by='simil').to_csv("FalseClaims_paths.csv",index=False)
	pos_neg=pd.concat([positive,negative],ignore_index=True)
	y=list(pos_neg['label'])
	scores=list(pos_neg['simil'])
	offset=len(positive)
	scores2=[]
	y2=[]
	try:
		for claimID in [*trueclaim_map]:
			claimRange=trueclaim_map[claimID]
			val=np.mean(scores[claimRange[0]:claimRange[1]])
			if not np.isnan(val):
				scores2.append(val)
				y2.append(int(np.mean(y[claimRange[0]:claimRange[1]])))
		for claimID in [*falseclaim_map]:
			claimRange=falseclaim_map[claimID]
			val=np.mean(scores[claimRange[0]:claimRange[1]])
			if not np.isnan(val):
				scores2.append(val)
				y2.append(int(np.mean(y[claimRange[0]:claimRange[1]])))
	except ValueError:
		print("Exception")
		pdb.set_trace()
	fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
	print(metrics.auc(fpr,tpr))
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr,tpr))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.title('ROC using Degree')
	plt.savefig("Exp2 ROC Using Degree")
	fpr, tpr, thresholds = metrics.roc_curve(y2, scores2, pos_label=1)
	print(metrics.auc(fpr,tpr))
	plt.figure()
	plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr,tpr))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.title('ROC using Degree and Claim Microaverage')
	plt.savefig("Exp2 ROC Using Degree and Claim Microaverage")

def plot_log(trueclaim_map,falseclaim_map):
	#klinker outputs json
	positive=pd.read_json("trueclaim_logdegree_u.json")
	positive['label']=1
	negative=pd.read_json("falseclaim_logdegree_u.json")
	negative['label']=0
	positive.filter(["simil","paths"]).sort_values(by='simil').to_csv("TrueClaims_paths_log.csv",index=False)
	negative.filter(["simil","paths"]).sort_values(by='simil').to_csv("FalseClaims_paths_log.csv",index=False)
	pos_neg=pd.concat([positive,negative],ignore_index=True)
	y=list(pos_neg['label'])
	scores=list(pos_neg['simil'])
	offset=len(positive)
	scores2=[]
	y2=[]
	try:
		for claimID in [*trueclaim_map]:
			claimRange=trueclaim_map[claimID]
			val=np.mean(scores[claimRange[0]:claimRange[1]])
			if not np.isnan(val):
				scores2.append(val)
				y2.append(int(np.mean(y[claimRange[0]:claimRange[1]])))
		for claimID in [*falseclaim_map]:
			claimRange=falseclaim_map[claimID]
			val=np.mean(scores[claimRange[0]:claimRange[1]])
			if not np.isnan(val):
				scores2.append(val)
				y2.append(int(np.mean(y[claimRange[0]:claimRange[1]])))
	except ValueError:
		print("Exception")
		pdb.set_trace()
	fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
	print(metrics.auc(fpr,tpr))
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr,tpr))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.title('ROC using Log Degree')
	plt.savefig("Exp2 ROC Using Log Degree")
	fpr, tpr, thresholds = metrics.roc_curve(y2, scores2, pos_label=1)
	print(metrics.auc(fpr,tpr))
	plt.figure()
	plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr,tpr))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.title('ROC using Log Degree and Claim Microaverage')
	plt.savefig("Exp2 ROC Using Log Degree and Claim Microaverage")

def plot_no():
	#klinker outputs json
	title='ROC using Log Degree No Overlap'
	positive=pd.read_json("trueclaim_logdegree_u_no.json")
	positive['label']=1
	negative=pd.read_json("falseclaim_logdegree_u_no.json")
	negative['label']=0
	positive.filter(["simil","paths"]).sort_values(by='simil').to_csv("TrueClaims_paths_log_no.csv",index=False)
	negative.filter(["simil","paths"]).sort_values(by='simil').to_csv("FalseClaims_paths_log_no.csv",index=False)
	pos_neg=pd.concat([positive,negative],ignore_index=True)
	y=list(pos_neg['label'])
	scores=list(pos_neg['simil'])
	offset=len(positive)
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

def overlap_triples(trueclaim_inputlist,falseclaim_inputlist):
	t1=len(trueclaim_inputlist)
	trueclaim_inputlist=set(map(str,list(map(list,trueclaim_inputlist.astype(int)))))
	t2=len(trueclaim_inputlist)
	f1=len(falseclaim_inputlist)
	falseclaim_inputlist=set(map(str,list(map(list,falseclaim_inputlist.astype(int)))))
	f2=len(falseclaim_inputlist)
	intersect=falseclaim_inputlist.intersection(trueclaim_inputlist)
	intersect_len=len(intersect)
	trueclaim_inputlist_no=trueclaim_inputlist-intersect
	falseclaim_inputlist_no=falseclaim_inputlist-intersect
	trueclaim_inputlist_no=pd.DataFrame(map(eval,list(trueclaim_inputlist_no))).replace(-9223372036854775808,np.nan)
	falseclaim_inputlist_no=pd.DataFrame(map(eval,list(falseclaim_inputlist_no))).replace(-9223372036854775808,np.nan)
	trueclaim_inputlist_new=[]
	falseclaim_inputlist_new=[]
	with codecs.open('trueclaim_triples_no.txt',"w","utf-8") as f:
		for i in range(len(trueclaim_inputlist_no)):
			line=np.asarray(trueclaim_inputlist_no.iloc[i])
			trueclaim_inputlist_new.append(line)
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2]),str(line[3]),str(int(line[4])),str(line[5]),str(line[6])))
	with codecs.open('falseclaim_triples_no.txt',"w","utf-8") as f:
		for i in range(len(falseclaim_inputlist_no)):
			line=np.asarray(falseclaim_inputlist_no.iloc[i])
			falseclaim_inputlist_new.append(line)
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2]),str(line[3]),str(int(line[4])),str(line[5]),str(line[6])))	
	np.save("trueclaim_triples_no.npy",trueclaim_inputlist_new)	
	np.save("falseclaim_triples_no.npy",falseclaim_inputlist_new)	
	
# uris,uris_dict,edgelist=parse_dbpedia()
# uris,uris_dict,trueclaim_uris,falseclaim_uris,trueclaim_map,falseclaim_map,trueclaim_inputlist,falseclaim_inputlist=load_stuff()
# trueclaim_inputlist,falseclaim_inputlist=load_stuff()
# overlap_triples(trueclaim_inputlist,falseclaim_inputlist)
# parse_claims(uris_dict)
# create_input_file(trueclaim_uris,falseclaim_uris)
calculate_DBPedia_stats()