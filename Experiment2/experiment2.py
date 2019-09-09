# -*- coding: utf-8 -*-
import rdflib
import os
import numpy as np
import json
import pandas as pd
import re
import pdb
import codecs
from decimal import Decimal
import networkx as nx 
from itertools import combinations,chain
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import pearsonr,kendalltau,spearmanr
from IPython.core.debugger import set_trace
import seaborn as sns
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

#Calculate Graph statistics of interest and saves them to a file called stats.txt


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

def read_pairs_fromfile_bulk_all():
	splits=28
	combined=pd.DataFrame()
	for i in range(splits+1):
		a=pd.read_json(os.path.join("DBPedia",str(i+1)+"_part_intersect_all_pairs_DBPedia_IDs_co.json"))
		print((i+1),len(a))
		combined=pd.concat([combined,a],ignore_index=True)
	print(len(combined))
	combined['label']="DBPedia"
	dbpedia_scores=list(combined['simil'])
	tfcg=pd.read_json(os.path.join("TFCG_co","Intersect_all_pairs_TFCG_co_IDs.json"))
	ffcg=pd.read_json(os.path.join("FFCG_co","Intersect_all_pairs_FFCG_co_IDs.json"))
	np.save(os.path.join("DBPedia","DBPedia_co_scores.npy"),dbpedia_scores)
	np.save(os.path.join("TFCG_co","TFCG_co_scores.npy"),list(tfcg['simil']))
	np.save(os.path.join("FFCG_co","FFCG_co_scores.npy"),list(ffcg['simil']))
	combined.to_csv("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph Data/DBPedia Data/Intersect_all_pairs_DBPedia_IDs_co.csv")

def quantile_correlations(removal,corr_type):
	tfcg_scores_all=list(np.load(os.path.join("TFCG_co","TFCG_co_scores.npy")))
	ffcg_scores_all=list(np.load(os.path.join("FFCG_co","FFCG_co_scores.npy")))
	dbpedia_scores_all=list(np.load(os.path.join("DBPedia","DBPedia_co_scores.npy")))
	title_text=""
	if removal:
		for name in ["tfcg_scores_all","ffcg_scores_all","dbpedia_scores_all"]:
			indices=[i for i, x in enumerate(eval(name)) if x == 0]
			for i in sorted(indices, reverse = True):
				del tfcg_scores_all[i]
				del ffcg_scores_all[i]
				del dbpedia_scores_all[i]
			title_text="_0removed"	
	scores_all=pd.DataFrame(columns=['DBPedia','TFCG','FFCG'])
	scores_all['DBPedia']=dbpedia_scores_all
	scores_all['TFCG']=tfcg_scores_all
	scores_all['FFCG']=ffcg_scores_all
	scores_all=scores_all.sort_values(by='DBPedia')
	scores_all=scores_all.reset_index(drop=True)
	percentiles=[0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05]
	dbpedia_percentiles=list(scores_all.quantile(percentiles)['DBPedia'])
	dataframes_percentiles=[scores_all[scores_all.DBPedia>i] for i in dbpedia_percentiles]
	correlations_percentile=pd.DataFrame(columns=['percentile','type',corr_type,'p-value','size'])
	for i in range(len(dataframes_percentiles)):
		dataframe=dataframes_percentiles[i]
		dbpedia_tfcg=eval(corr_type)(dataframe['DBPedia'],dataframe['TFCG'])
		dbpedia_ffcg=eval(corr_type)(dataframe['DBPedia'],dataframe['FFCG'])
		tfcg_ffcg=eval(corr_type)(dataframe['TFCG'],dataframe['FFCG'])
		dbpedia_tfcg_row={'percentile':percentiles[i],'type':'DBPedia - TFCG',corr_type:dbpedia_tfcg[0],'p-value':pvalue_sign(dbpedia_tfcg[1]),'size':len(dataframe)}
		dbpedia_ffcg_row={'percentile':percentiles[i],'type':'DBPedia - FFCG',corr_type:dbpedia_ffcg[0],'p-value':pvalue_sign(dbpedia_ffcg[1]),'size':len(dataframe)}
		# tfcg_ffcg_row={'percentile':percentiles[i],'type':'TFCG - FFCG',corr_type:tfcg_ffcg[0],'p-value':tfcg_ffcg[1],'size':len(dataframe)}
		correlations_percentile=correlations_percentile.append(dbpedia_tfcg_row, ignore_index=True)
		correlations_percentile=correlations_percentile.append(dbpedia_ffcg_row, ignore_index=True)
		# correlations_percentile=correlations_percentile.append(tfcg_ffcg_row, ignore_index=True)
	correlations_percentile_dbpedia_tfcg=correlations_percentile[correlations_percentile['type']=="DBPedia - TFCG"].reset_index()
	correlations_percentile_dbpedia_ffcg=correlations_percentile[correlations_percentile['type']=="DBPedia - FFCG"].reset_index().reset_index()
	correlations_percentile_tfcg_ffcg=correlations_percentile[correlations_percentile['type']=="TFCG - FFCG"].reset_index()
	title="Percentile Correlations Co-Occur"
	plt.figure(1)
	plt.title("Co-Occur Network "+corr_type+title_text)
	plt.plot(percentiles,correlations_percentile_dbpedia_tfcg[corr_type],label="DBPedia - TFCG",marker='o',linestyle='dashed')
	for i,value in enumerate(correlations_percentile_dbpedia_tfcg['p-value']):
		plt.annotate(value,(percentiles[i],correlations_percentile_dbpedia_tfcg[corr_type][i]))
	plt.plot(percentiles,correlations_percentile_dbpedia_ffcg[corr_type],label="DBPedia - FFCG",marker='o',linestyle='dashed')
	for i,value in enumerate(correlations_percentile_dbpedia_ffcg['p-value']):
		plt.annotate(value,(percentiles[i],correlations_percentile_dbpedia_ffcg[corr_type][i]))	
	# plt.plot(percentiles,correlations_percentile[correlations_percentile['type']=="TFCG - FFCG"][corr_type],label="TFCG - FFCG")
	plt.xlabel("Decreasing Proximity Percentiles")
	plt.gca().invert_xaxis()
	plt.ylabel("Correlations")
	plt.legend(loc="lower left")
	plt.savefig(title.replace(" ","_")+title_text+"_"+corr_type+".png")
	plt.close()
	plt.clf()

def pvalue_sign(pvalue):
	if pvalue>0.01:
		return '!'
	else:
		return '*'

def correlations(removal):
	tfcg_scores_all=list(np.load(os.path.join("TFCG_co","TFCG_co_scores.npy")))
	ffcg_scores_all=list(np.load(os.path.join("FFCG_co","FFCG_co_scores.npy")))
	dbpedia_scores_all=list(np.load(os.path.join("DBPedia","DBPedia_co_scores.npy")))
	print("TFCG_co-DBPedia_co",kendalltau(tfcg_scores_all,dbpedia_scores_all))
	print("FFCG_co-DBPedia_co",kendalltau(ffcg_scores_all,dbpedia_scores_all))
	print("TFCG_co-FFCG_co",kendalltau(tfcg_scores_all,ffcg_scores_all))
	title="Proximity Distribution"
	plt.figure(1)
	plt.hist(tfcg_scores_all,histtype='step',label='TFCG')
	plt.hist(ffcg_scores_all,histtype='step',label='FFCG')
	plt.hist(dbpedia_scores_all,histtype='step',label='DBPedia')
	plt.xlabel("Proximity Scores")
	plt.ylabel("Density")
	plt.legend(loc="upper right")
	plt.savefig(title.replace(" ","_")+".png")
	plt.close()
	plt.clf()
	sns.set_style("white")
	sns.set_style("ticks")
	ax = sns.kdeplot(pd.Series(tfcg_scores_all,name="TFCG"))
	ax = sns.kdeplot(pd.Series(ffcg_scores_all,name="FFCG"))
	ax = sns.kdeplot(pd.Series(dbpedia_scores_all,name="DBPedia"))
	plt.xlabel("Proximity Scores")
	plt.ylabel("Density")
	plt.savefig(title.replace(" ","_")+"_seaborn.png")
	plt.close()
	plt.clf()
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
	print("TFCG_co-DBPedia_co",kendalltau(tfcg_scores_all,dbpedia_scores_all))
	print("FFCG_co-DBPedia_co",kendalltau(ffcg_scores_all,dbpedia_scores_all))
	print("TFCG_co-FFCG_co",kendalltau(tfcg_scores_all,ffcg_scores_all))
	title="Proximity Distribution"
	plt.figure(2)
	# plt.hist(tfcg_scores_all,histtype='step',label='TFCG')
	# plt.hist(ffcg_scores_all,histtype='step',label='FFCG')
	# plt.hist(dbpedia_scores_all,histtype='step',label='DBPedia')
	# plt.xlabel("Proximity Scores Co-Occur")
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
	plt.title("Co-Occur Network "+title_text)
	plt.ylabel("Density")
	plt.savefig(title.replace(" ","_")+title_text+".png")
	plt.close()
	plt.clf()


def testlengths():
	data=pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/claimreviews_db2.csv",index_col=0)
	##Dropping non-str rows
	filter=list(map(lambda x:type(x)!=str,data['rating_name']))
	data.drop(data[filter].index,inplace=True)
	print(data.groupby('fact_checkerID').count())
	trueregex=re.compile(r'(?i)^true|^correct$|^mostly true$|^geppetto checkmark$')
	falseregex=re.compile(r'(?i)^false|^mostly false|^pants on fire$|^four pinocchios$|^no\ |^no:|^distorts the facts|^wrong$')
	trueind=data['rating_name'].apply(lambda x:trueregex.match(x)!=None)
	trueclaims=list(data.loc[trueind]['claim_text'])
	falseind=data['rating_name'].apply(lambda x:falseregex.match(x)!=None)
	falseclaims=list(data.loc[falseind]['claim_text'])
	np.save("true_claims.npy",list(trueclaims))
	np.save("false_claims.npy",list(falseclaims))
	trueclaims=np.load("true_claims.npy")
	falseclaims=np.load("false_claims.npy")
	trueclaims=[claim.split() for claim in trueclaims]
	falseclaims=[claim.split() for claim in falseclaims]
	trueclaims_lengths=list(map(len,trueclaims))
	falseclaims_lengths=list(map(len,falseclaims))
	title="Claim Length Distribution"
	plt.figure(1)
	sns.set_style("white")
	sns.set_style("ticks")
	ax = sns.kdeplot(pd.Series(trueclaims_lengths,name="True Claims Mean {} Median {}".format(round(np.mean(trueclaims_lengths),2),int(np.median(trueclaims_lengths)))))
	ax = sns.kdeplot(pd.Series(falseclaims_lengths,name="False Claims Mean {} Median {}".format(round(np.mean(falseclaims_lengths),2),int(np.median(falseclaims_lengths)))))
	plt.title(title+" P-Value %.2E" %Decimal(stats.ttest_ind(trueclaims_lengths,falseclaims_lengths).pvalue))
	plt.xlabel("Claim Length")
	plt.legend(loc="upper right")
	plt.ylabel("Density")
	plt.savefig(title.replace(" ","_")+"_seaborn.png")
	plt.close()
	plt.clf()

def create_cooccurrence_network():
	# data=pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/claimreviews_db2.csv",index_col=0)
	# ##Dropping non-str rows
	# filter=list(map(lambda x:type(x)!=str,data['rating_name']))
	# data.drop(data[filter].index,inplace=True)
	# print(data.groupby('fact_checkerID').count())
	# trueregex=re.compile(r'(?i)^true|^correct$|^mostly true$|^geppetto checkmark$')
	# falseregex=re.compile(r'(?i)^false|^mostly false|^pants on fire$|^four pinocchios$|^no\ |^no:|^distorts the facts|^wrong$')
	# trueind=data['rating_name'].apply(lambda x:trueregex.match(x)!=None)
	# trueclaims=list(data.loc[trueind]['claimID'])
	# falseind=data['rating_name'].apply(lambda x:falseregex.match(x)!=None)
	# falseclaims=list(data.loc[falseind]['claimID'])
	# np.save("true_claimID_list.npy",list(trueclaims))
	# np.save("false_claimID_list.npy",list(falseclaims))
	# trueclaims=np.load("true_claimID_list.npy")
	# falseclaims=np.load("false_claimID_list.npy")
	# trueclaim_uris={}
	# trueclaim_edges={}
	# falseclaim_uris={}
	# falseclaim_edges={}
	# TFCG_co=nx.Graph()
	# FFCG_co=nx.Graph()
	# dbpediaregex=re.compile(r'http:\/\/dbpedia\.org\/resource\/')
	# for t in trueclaims:
	# 	claim_uris=set([])
	# 	g=rdflib.Graph()
	# 	filename="claim"+str(t)+".rdf"
	# 	try:
	# 		g.parse("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/"+filename,format='application/rdf+xml')
	# 	except:
	# 		# continue
	# 		pass
	# 	for triple in g:
	# 		subject,predicate,obj=list(map(str,triple))
	# 		try:
	# 			if dbpediaregex.search(subject):
	# 				claim_uris.add(subject)
	# 			if dbpediaregex.search(obj):
	# 				claim_uris.add(obj)
	# 		except KeyError:
	# 			continue
	# 	trueclaim_uris[t]=list(claim_uris)
	# 	trueclaim_edges[t]=list(combinations(trueclaim_uris[t],2))
	# 	TFCG_co.add_edges_from(trueclaim_edges[t])
	# nx.write_edgelist(TFCG_co,'TFCG_co.edgelist',data=False)
	# for f in falseclaims:
	# 	claim_uris=set([])
	# 	g=rdflib.Graph()
	# 	filename="claim"+str(f)+".rdf"
	# 	try:
	# 		g.parse("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/"+filename,format='application/rdf+xml')
	# 	except:
	# 		# continue
	# 		pass
	# 	for triple in g:
	# 		subject,predicate,obj=list(map(str,triple))
	# 		try:
	# 			if dbpediaregex.search(subject):
	# 				claim_uris.add(subject)
	# 			if dbpediaregex.search(obj):
	# 				claim_uris.add(obj)
	# 		except KeyError:
	# 			continue
	# 	falseclaim_uris[f]=list(claim_uris)
	# 	falseclaim_edges[f]=list(combinations(falseclaim_uris[f],2))
	# 	FFCG_co.add_edges_from(falseclaim_edges[f])
	# nx.write_edgelist(FFCG_co,'FFCG_co.edgelist',data=False)
	TFCG_co=nx.read_edgelist(os.path.join("TFCG_co","TFCG_co.edgelist")) 
	FFCG_co=nx.read_edgelist(os.path.join("FFCG_co","FFCG_co.edgelist")) 
	for mode in ["TFCG_co","FFCG_co"]:
		uris=list(eval(mode).nodes())
		edges=list(eval(mode).edges())
		#Saving the uris
		with codecs.open(os.path.join(mode,mode+"_uris.txt"),"w","utf-8") as f:
			np.save(os.path.join(mode,mode+"_uris.npy"),uris)
			for uri in uris:
				try:
					f.write(str(uri)+"\n")
				except:
					pdb.set_trace()
		uris_dict={uris[i]:i for i in range(len(uris))}
		#Saving the dictionaries
		with codecs.open(os.path.join(mode,mode+"_uris_dict.json"),"w","utf-8") as f:
			f.write(json.dumps(uris_dict,ensure_ascii=False))
		#Saving the edgelist to input Knowledge Linker
		edgelist=np.asarray([[uris_dict[edge[0]],uris_dict[edge[1]],1] for edge in edges])
		np.save(os.path.join(mode,mode+"_edgelist.npy"),edgelist)
	set_trace()
	#Loading the dictionaries
	with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph Data/DBPedia Data/dbpedia_uris_dict.json","r","utf-8") as f:
		DBPedia_uris_dict=json.loads(f.read())
	with codecs.open("TFCG_co/TFCG_co_uris_dict.json","r","utf-8") as f:
		TFCG_co_uris_dict=json.loads(f.read())
	with codecs.open("FFCG_co/FFCG_co_uris_dict.json","r","utf-8") as f:
		FFCG_co_uris_dict=json.loads(f.read())
	intersect_uris=np.asarray(list(set(TFCG_co.nodes()).intersection(set(FFCG_co.nodes()))))
	intersect_uris=np.asarray(list(set(intersect_uris).intersection(set(DBPedia_uris_dict.keys()))))
	np.save("intersect_dbpedia_uris_co.npy",list(intersect_uris))
	#Loading True Pairs after DBPedia has found the adjacent nodes
	# intersect_true_pairs=np.load("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/intersect_entity_true_pairs_dbpedia_co.npy")
	# #Converting list to set, to get rid of duplicates
	# intersect_true_pairs_set=set(list(map(str,list(map(set,intersect_true_pairs)))))
	# set_trace()
	# #Converting it back. Getting rid of pairs where both uris are duplicates, as well as duplicate of each pair
	# intersect_true_pairs=np.asarray([i for i in list(map(list,list(map(eval,list(intersect_true_pairs_set))))) if len(i)==2])
	#Find all possible combinations of these uris
	intersect_all_pairs=combinations(intersect_uris,2)
	intersect_all_pairs=np.asarray(list(map(list,intersect_all_pairs)))
	# set_trace()
	#Choosing 2n random pairs, where n is the lenght of the total true_pairs 
	# random_pairs=np.random.choice(range(len(intersect_all_pairs)),size=len(intersect_true_pairs)*2,replace=False)
	# intersect_false_pairs=[]
	# rejected_pairs=[]
	# set_trace()
	# #Rejecting pairs from random pairs that are already present in true pairs
	# counter=0
	# for i in random_pairs:
	# 	if counter<len(intersect_true_pairs):
	# 		if str(set(intersect_all_pairs[i])) in intersect_true_pairs_set or str(set(list(reversed(intersect_all_pairs[i])))) in intersect_true_pairs_set:#eliminating random triple if it exists in the intersect set (converted individiual triples to str to make a set)
	# 			rejected_pairs.append(intersect_all_pairs[i])
	# 		else:
	# 			counter+=1
	# 			intersect_false_pairs.append(intersect_all_pairs[i])
	# 	else:
	# 		break
	# intersect_true_pairs=np.asarray(intersect_true_pairs)
	# intersect_false_pairs=np.asarray(intersect_false_pairs)
	# #Saving true and false pairs pairs
	# np.save("intersect_true_pairs_co.npy",intersect_true_pairs)
	# np.save("intersect_false_pairs_co.npy",intersect_false_pairs)
	# #Loading saved True Pairs
	# intersect_true_pairs=np.load("intersect_true_pairs_co.npy")
	# intersect_false_pairs=np.load("intersect_false_pairs_co.npy")
	# # Reformatting according to the input format acccepted by Knowledge Linker
	# intersect_true_pairs=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect_true_pairs])
	# intersect_false_pairs=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect_false_pairs])
	######################################################################Writing True Pairs
	# Writing true pairs to file using TFCG,FFCG and DBPedia entity IDs
	# for mode in ["TFCG_co","FFCG_co","DBPedia"]:
	# 	with codecs.open(os.path.join(mode,'Intersect_true_pairs_'+mode+'_IDs.txt'),"w","utf-8") as f:
	# 		for line in intersect_true_pairs:
	# 			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(eval(mode+"_uris_dict")[line[1]])),str(line[2]),str(line[3]),str(int(eval(mode+"_uris_dict")[line[4]])),str(line[5]),str(line[6])))
	# 	with codecs.open(os.path.join(mode,'Intersect_false_pairs_'+mode+'_IDs.txt'),"w","utf-8") as f:
	# 		for line in intersect_false_pairs:
	# 			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(eval(mode+"_uris_dict")[line[1]])),str(line[2]),str(line[3]),str(int(eval(mode+"_uris_dict")[line[4]])),str(line[5]),str(line[6])))
	intersect_all_pairs=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in intersect_all_pairs])
	for mode in ["TFCG_co","FFCG_co"]:
		with codecs.open(os.path.join(mode,'Intersect_all_pairs_'+mode+'_IDs.txt'),"w","utf-8") as f:
			for line in intersect_all_pairs:
				f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(eval(mode+"_uris_dict")[line[1]])),str(line[2]),str(line[3]),str(int(eval(mode+"_uris_dict")[line[4]])),str(line[5]),str(line[6])))
	splits=28
	hours=30
	partition=int(len(intersect_all_pairs)/splits)
	for i in range(0,splits):	
		with codecs.open(os.path.join("DBPedia",str(i+1)+'_part_intersect_all_pairs_DBPedia_IDs_co.txt'),"w","utf-8") as f:
			for line in intersect_all_pairs[partition*i:partition*(i+1)]:
				f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))
		with codecs.open(os.path.join("DBPedia",str(i+1)+'_job_co.sh'),"w","utf-8") as f:
			f.write('''
#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=180gb,walltime={}:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N {}_KLinker
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/DBPedia
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_edgelist.npy {}_part_intersect_all_pairs_DBPedia_IDs_co.txt {}_part_intersect_all_pairs_DBPedia_IDs_co.json -u -n 12
				'''.format(hours,i+1,i+1,i+1))
	with codecs.open(os.path.join("DBPedia",str(splits+1)+'_part_intersect_all_pairs_DBPedia_IDs_co.txt'),"w","utf-8") as f:
		for line in intersect_all_pairs[splits*partition:]:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(DBPedia_uris_dict[line[1]])),str(line[2]),str(line[3]),str(int(DBPedia_uris_dict[line[4]])),str(line[5]),str(line[6])))
	with codecs.open(os.path.join("DBPedia",str(splits+1)+'_job_co.sh'),"w","utf-8") as f:
		f.write('''
#PBS -k o
#PBS -l nodes=1:ppn=12,vmem=180gb,walltime={}:00:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N {}_KLinker
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate env-kl
cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/DBPedia
time klinker linkpred /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_uris.txt /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph\\ Data/DBPedia\\ Data/dbpedia_edgelist.npy {}_part_intersect_all_pairs_DBPedia_IDs_co.txt {}_part_intersect_all_pairs_DBPedia_IDs_co.json -u -n 12
			'''.format(hours,splits+1,splits+1,splits+1))


	# with codecs.open("trueclaim_uris.json","w","utf-8") as f:
	# 	f.write(json.dumps(trueclaim_uris,ensure_ascii=False))
	# with codecs.open("falseclaim_uris.json","w","utf-8") as f:
	# 	f.write(json.dumps(falseclaim_uris,ensure_ascii=False))
	# with codecs.open("trueclaim_edges.json","w","utf-8") as f:
	# 	f.write(json.dumps(trueclaim_edges,ensure_ascii=False))
	# with codecs.open("falseclaim_edges.json","w","utf-8") as f:
	# 	f.write(json.dumps(falseclaim_edges,ensure_ascii=False))

def plot_TFCGvsFFCG():
	Intersect_true_TFCG=pd.read_json("TFCG_co/Intersect_true_pairs_TFCG_co_IDs.json")
	Intersect_true_FFCG=pd.read_json("FFCG_co/Intersect_true_pairs_FFCG_co_IDs.json")
	Intersect_true_DBPedia=pd.read_json("DBPedia/Intersect_true_pairs_DBPedia_IDs.json")

	Intersect_false_TFCG=pd.read_json("TFCG_co/Intersect_false_pairs_TFCG_co_IDs.json")
	Intersect_false_FFCG=pd.read_json("FFCG_co/Intersect_false_pairs_FFCG_co_IDs.json")
	Intersect_false_DBPedia=pd.read_json("DBPedia/Intersect_false_pairs_DBPedia_IDs.json")


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
	plt.figure(1)
	####TFCG
	fpr, tpr, thresholds = metrics.roc_curve(y_TFCG, scores_TFCG, pos_label=1)
	print(metrics.auc(fpr,tpr))
	print("TFCG P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_true_TFCG['simil'],Intersect_false_TFCG['simil']).pvalue))
	plt.plot(fpr, tpr,lw=lw, label='TFCG (AUC = %0.2f) ' % metrics.auc(fpr,tpr))
	####FFCG
	fpr, tpr, thresholds = metrics.roc_curve(y_FFCG, scores_FFCG, pos_label=1)
	print(metrics.auc(fpr,tpr))
	print("FFCG P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_true_FFCG['simil'],Intersect_false_FFCG['simil']).pvalue))
	plt.plot(fpr, tpr,lw=lw, label='FFCG (AUC = %0.2f) ' % metrics.auc(fpr,tpr))
	####DBPedia
	fpr, tpr, thresholds = metrics.roc_curve(y_DBPedia, scores_DBPedia, pos_label=1)
	print(metrics.auc(fpr,tpr))
	print("DBPedia P-Value %.2E" %Decimal(stats.ttest_rel(Intersect_true_DBPedia['simil'],Intersect_false_DBPedia['simil']).pvalue))
	plt.plot(fpr, tpr,lw=lw, label='DBPedia (AUC = %0.2f) ' % metrics.auc(fpr,tpr))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, label='Baseline',linestyle='--')
	plt.legend(loc="lower right")
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.savefig(title.replace(" ","_")+".png")
	plt.close()
	plt.clf()
	##########################False2
	# plt.figure(2)
	# ####TFCG
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
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.savefig(title.replace(" ","_")+"2.png")
	# plt.close()
	# plt.clf()
	##########################Distribution Plot
	#Plot 3
	# plt.figure(3)
	# title="Score Difference between True and False Pairs"
	# #first overlap plot TFCG vs FFCG
	# intersect=Intersect_true_TFCG['simil']
	# intersect=intersect.reset_index(drop=True)
	# set_trace()
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
#Calculate Graph statistics of interest and saves them to a file called stats.txt
def calculate_stats():
	degreelist={}
	pathlengths={}
	for mode in ['TFCG_co','FFCG_co']:
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
def calculate_claim_stats(trueclaim_uris,falseclaim_uris,trueclaim_inputlist,falseclaim_inputlist):
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
# calculate_DBPedia_stats()