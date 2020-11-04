import os
import pandas as pd
import re
import numpy as np
import argparse
import rdflib
import codecs
import json
import seaborn as sns
from sklearn import metrics
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
import sys
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
# sys.path.insert(1, '/geode2/home/u110/zkachwal/BigRed3/factcheckgraph/create_graphs')
# from create_fred import *
# from create_co_occur import *
import csv

def chunkstring(string,length):
	chunks=int(len(string)/length)+1
	offset=0
	for i in range(1,chunks):
		if i*length<len(string):
			if string[i*length]==' ':
				loc=i*length
			else:
				left=string[:i*length][::-1].find(" ")
				right=string[i*length:].find(" ")
				if left<right:
					loc=i*length-left-1
				else:
					loc=i*length+right
			string=string[:loc]+"\n"+string[loc+1:]
	return string.splitlines()

def find_baseline(rdf_path,graph_path,model_path,embed_path,graph_type,fcg_class,cpu):
	'''
	To have an accurate baseline for future: 1. remove claims that do no exist in the fred graph
	2. Remove claims that are near duplicates
	'''
	#load true and false claims
	true_claims_embed=pd.read_csv(os.path.join(embed_path,"true_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	true_claims_labels=pd.read_csv(os.path.join(embed_path,"true_claims_embeddings_labels_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	false_claims_embed=pd.read_csv(os.path.join(embed_path,"false_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	false_claims_labels=pd.read_csv(os.path.join(embed_path,"false_claims_embeddings_labels_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	# if the baseline is for a directed fcg, we only choose the claimIDs used for finding paths in the directed fcg
	if graph_type=='directed':
		true_claimIDs=np.array(list(map(int,true_claims_labels[1:,1])))
		false_claimIDs=np.array(list(map(int,false_claims_labels[1:,1])))
		directed_tfcg_claimIDs=np.load(os.path.join(graph_path,fcg_class,'tfcg','directed_tfcg_true_claimIDs.npy'))
		directed_ffcg_claimIDs=np.load(os.path.join(graph_path,fcg_class,'tfcg','directed_tfcg_false_claimIDs.npy'))
		true_claimIDs=np.nonzero(np.isin(true_claimIDs,directed_tfcg_claimIDs,assume_unique=True))[0]
		false_claimIDs=np.nonzero(np.isin(false_claimIDs,directed_ffcg_claimIDs,assume_unique=True))[0]
		true_claims_embed=np.take(true_claims_embed,true_claimIDs,axis=0)
		false_claims_embed=np.take(false_claims_embed,true_claimIDs,axis=0)
	#Calculating angular distance
	true_true=1-np.arccos(np.clip(cosine_similarity(true_claims_embed,true_claims_embed),-1,1))/np.pi
	true_false=1-np.arccos(np.clip(cosine_similarity(true_claims_embed,false_claims_embed),-1,1))/np.pi
	false_false=1-np.arccos(np.clip(cosine_similarity(false_claims_embed,false_claims_embed),-1,1))/np.pi
	np.fill_diagonal(true_true,np.nan)
	np.fill_diagonal(false_false,np.nan)
	##################################################################################ROC PLOT
	plt.figure(figsize=(9, 8))
	lw=2
	plt.plot([0, 1], [0, 1],color='navy',lw=lw,linestyle='--')
	true0_y=[0 for i in range(len(true_true))]
	true1_y=[1 for i in range(len(true_true))]
	false0_y=[0 for i in range(len(false_false))]
	false1_y=[1 for i in range(len(false_false))]

	title="ROC Baseline using angular similarity of claims"

	true_true_mean=np.apply_along_axis(np.nanmean,1,true_true)
	true_false_true_mean=np.apply_along_axis(np.nanmean,1,true_false)
	false_false_mean=np.apply_along_axis(np.nanmean,1,false_false)
	true_false_false_mean=np.apply_along_axis(np.nanmean,0,true_false)
	# Zoher's formula
	# true_scores=(true_true_mean-true_false_true_mean)/((true_true_mean*len(true_true)+true_false_true_mean*len(false_false))/(len(true_true)+len(false_false)))
	# false_scores=(true_false_false_mean-false_false_mean)/((true_false_false_mean*len(true_true)+false_false_mean*len(false_false))/(len(true_true)+len(false_false)))
	# Fil's formula
	true_scores=list((true_true_mean*len(true_true)-true_false_true_mean*len(false_false))/(true_true_mean*len(true_true)+true_false_true_mean*len(false_false)))
	false_scores=list((true_false_false_mean*len(true_true)-false_false_mean*len(false_false))/(true_false_false_mean*len(true_true)+false_false_mean*len(false_false)))
	#
	fpr,tpr,thresholds=metrics.roc_curve(true1_y+false0_y,list(true_scores)+list(false_scores), pos_label=1)
	plt.plot(fpr,tpr,lw=lw,label='scores (%0.2f) '%metrics.auc(fpr,tpr))

	# fpr,tpr,thresholds=metrics.roc_curve(true1_y+false0_y,list(true_false_true_mean)+list(false_false_mean), pos_label=1)
	# plt.plot(fpr,tpr,lw=lw,label='false claims mean (%0.2f) '%metrics.auc(fpr,tpr))

	# true_true_min=np.apply_along_axis(np.nanmin,1,true_true)
	# true_false_true_min=np.apply_along_axis(np.nanmin,1,true_false)
	# false_false_min=np.apply_along_axis(np.nanmin,1,false_false)
	# true_false_false_min=np.apply_along_axis(np.nanmin,0,true_false)

	# fpr,tpr,thresholds=metrics.roc_curve(true0_y+false1_y,list(true_true_min)+list(true_false_false_min), pos_label=1)
	# plt.plot(fpr,tpr,lw=lw,label='true claims min (%0.2f) '%metrics.auc(fpr,tpr))

	# fpr,tpr,thresholds=metrics.roc_curve(true1_y+false0_y,list(true_false_true_min)+list(false_false_min), pos_label=1)
	# plt.plot(fpr,tpr,lw=lw,label='false claims min (%0.2f) '%metrics.auc(fpr,tpr))

	# true_true_max=np.apply_along_axis(np.nanmax,1,true_true)
	# true_false_true_max=np.apply_along_axis(np.nanmax,1,true_false)
	# false_false_max=np.apply_along_axis(np.nanmax,1,false_false)
	# true_false_false_max=np.apply_along_axis(np.nanmax,0,true_false)

	# fpr,tpr,thresholds=metrics.roc_curve(true0_y+false1_y,list(true_true_max)+list(true_false_false_max), pos_label=1)
	# plt.plot(fpr,tpr,lw=lw,label='true claims max (%0.2f) '%metrics.auc(fpr,tpr))

	# fpr,tpr,thresholds=metrics.roc_curve(true1_y+false0_y,list(true_false_true_max)+list(false_false_max), pos_label=1)
	# plt.plot(fpr,tpr,lw=lw,label='false claims max (%0.2f) '%metrics.auc(fpr,tpr))

	plt.xlabel('True Positive Rate')
	plt.ylabel('False Positive Rate')
	plt.legend(loc="lower right")
	plt.title(title)
	plt.tight_layout()
	x = datetime.datetime.now().strftime("%c")
	title+=" "+x
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()
	title=title.replace('ROC','KDE')
	##################################################################################KDE PLOT
	plt.figure(figsize=(9, 8))
	minscore=np.min(true_scores+false_scores)
	maxscore=np.max(true_scores+false_scores)
	intervalscore=float(maxscore-minscore)/20
	print(intervalscore)
	print(minscore)
	print(maxscore)
	sns.distplot(true_scores,hist=True,kde=True,bins=np.arange(minscore,maxscore+intervalscore,intervalscore),kde_kws={'linewidth': 3},label="true",norm_hist=True)
	sns.distplot(false_scores,hist=True,kde=True,bins=np.arange(minscore,maxscore+intervalscore,intervalscore),kde_kws={'linewidth': 3},label="false",norm_hist=True)
	plt.xlabel('similarity Scores (higher is positive)')
	plt.ylabel('Density')
	plt.legend(loc="upper right")
	plt.title(title)
	plt.tight_layout()
	x = datetime.datetime.now().strftime("%c")
	title+=" "+x
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()

def find_knn(rdf_path,graph_path,model_path,embed_path,graph_type,fcg_class,cpu,n_neighbors):
	'''
	To have an accurate baseline for future: 1. remove claims that do no exist in the fred graph
	2. Remove claims that are near duplicates
	'''
	#load true and false claims
	true_claims_embed=pd.read_csv(os.path.join(embed_path,"true_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	true_claims_labels=pd.read_csv(os.path.join(embed_path,"true_claims_embeddings_labels_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	false_claims_embed=pd.read_csv(os.path.join(embed_path,"false_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	false_claims_labels=pd.read_csv(os.path.join(embed_path,"false_claims_embeddings_labels_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	# if the baseline is for a directed fcg, we only choose the claimIDs used for finding paths in the directed fcg
	if graph_type=='directed':
		true_claimIDs=np.array(list(map(int,true_claims_labels[1:,1])))
		false_claimIDs=np.array(list(map(int,false_claims_labels[1:,1])))
		directed_tfcg_claimIDs=np.load(os.path.join(graph_path,fcg_class,'tfcg','directed_tfcg_true_claimIDs.npy'))
		directed_ffcg_claimIDs=np.load(os.path.join(graph_path,fcg_class,'tfcg','directed_tfcg_false_claimIDs.npy'))
		true_claimIDs=np.nonzero(np.isin(true_claimIDs,directed_tfcg_claimIDs,assume_unique=True))[0]
		false_claimIDs=np.nonzero(np.isin(false_claimIDs,directed_ffcg_claimIDs,assume_unique=True))[0]
		true_claims_embed=np.take(true_claims_embed,true_claimIDs,axis=0)
		false_claims_embed=np.take(false_claims_embed,true_claimIDs,axis=0)
	#set y labels
	true_y=[1 for i in range(len(true_claims_embed))]
	false_y=[-1 for i in range(len(false_claims_embed))]
	Y=np.array(true_y+false_y)
	#set x matrix
	X=np.concatenate((true_claims_embed,false_claims_embed),axis=0)
	#find pairwise cosine similarity
	X=np.arccos(np.clip(cosine_similarity(X,X),-1,1))/np.pi
	# data from claim labels: text, rating, claimID
	Xlabels=np.concatenate((true_claims_labels[1:],false_claims_labels[1:]),axis=0)
	np.fill_diagonal(X,np.nan)
	#Finding n_neighbors
	Xn=np.argpartition(X,n_neighbors)[:,:n_neighbors]
	# Yh=np.apply_along_axis(lambda x:np.mean(Y[x]),1,Xn)
	#Intializing Yh to 0
	Yh=np.array([float(0) for i in range(Xn.shape[0])])
	#text in Xlabels
	Xlabel=np.apply_along_axis(lambda x:Xlabels[x,0],1,Xn)
	#claimID in Xlabels
	XclaimID=np.apply_along_axis(lambda x:Xlabels[x,1],1,Xn)
	#rating in Xlabels
	Xrating=np.apply_along_axis(lambda x:Xlabels[x,2],1,Xn)
	Xndict={}
	for i in range(Xn.shape[0]):
		text=chunkstring(Xlabels[i][0],100)
		Xndict[i]={}
		aggregate=0
		for j in range(Xn.shape[1]):
			Xndict[i][int(Xn[i,j])]={}
			n_text=chunkstring(Xlabels[Xn[i,j]][0],100)
			Xndict[i][Xn[i,j]]={'n_text':n_text,'claimID':XclaimID[i,j],'rating':Xrating[i,j],'dist':round(X[i,Xn[i,j]],2)}
			aggregate+=Y[Xn[i,j]]*(1-X[i,Xn[i,j]])
		Xndict[i]=OrderedDict(sorted(Xndict[i].items(), key=lambda t: t[1]['dist']))
		Yh[i]=float(aggregate)/n_neighbors
		Xndict[i]['text']=text
		Xndict[i]['predscore']=round(Yh[i],2)
		Xndict[i]['score']=float(Y[i])
	#True claims (label=1)
	Xndict_1={t[0]:t[1] for t in Xndict.items() if t[1]['score']==1}
	#False claims (label=-1)
	Xndict_0={t[0]:t[1] for t in Xndict.items() if t[1]['score']==-1}
	#False claims with high predscore
	
	Xndict_1=OrderedDict(sorted(Xndict_1.items(), key=lambda t:np.mean([x['dist'] for x in t[1].items() if type(x)==dict])))
	Xndict_0=OrderedDict(sorted(Xndict_0.items(), key=lambda t:np.mean([x['dist'] for x in t[1].items() if type(x)==dict])))
	with codecs.open(os.path.join(embed_path,"knn_1_{}.json".format(n_neighbors)),"w","utf-8") as f:
		f.write(json.dumps(Xndict_1,indent=5,ensure_ascii=False))
	with codecs.open(os.path.join(embed_path,"knn_0_{}.json".format(n_neighbors)),"w","utf-8") as f:
		f.write(json.dumps(Xndict_0,indent=5,ensure_ascii=False))

	Xndict_0_high={t[0]:t[1] for t in Xndict.items() if t[1]['score']==-1 and t[1]['predscore']>0}
	Xndict_0_high=OrderedDict(sorted(Xndict_0_high.items(), key=lambda t:t[1]['predscore'],reverse=True))
	with codecs.open(os.path.join(embed_path,"knn_0_high_{}.json".format(n_neighbors)),"w","utf-8") as f:
		f.write(json.dumps(Xndict_0,indent=5,ensure_ascii=False))
	##################################################################################ROC PLOT
	plt.figure(figsize=(9, 8))
	lw=2
	plt.plot([0, 1], [0, 1],color='navy',lw=lw,linestyle='--')
	title="ROC KNN using angular similarity of claims K={}".format(n_neighbors)
	fpr,tpr,thresholds=metrics.roc_curve(Y,Yh,pos_label=1)
	plt.plot(fpr,tpr,lw=lw,label='scores (%0.2f) '%metrics.auc(fpr,tpr))
	plt.xlabel('True Positive Rate')
	plt.ylabel('False Positive Rate')
	plt.legend(loc="lower right")
	plt.title(title)
	plt.tight_layout()
	x = datetime.datetime.now().strftime("%c")
	title+=" "+x
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()
	##################################################################################KDE PLOT
	title=title.replace('ROC','KDE')
	plt.figure(figsize=(9, 8))
	minscore=np.min(list(Yh[Y==1])+list(Yh[Y==-1]))
	maxscore=np.max(list(Yh[Y==1])+list(Yh[Y==-1]))
	intervalscore=float(maxscore-minscore)/20
	print(intervalscore)
	print(minscore)
	print(maxscore)
	sns.distplot(Yh[Y==1],hist=True,kde=True,bins=np.arange(minscore,maxscore+intervalscore,intervalscore),kde_kws={'linewidth': 3},label="true",norm_hist=True)
	sns.distplot(Yh[Y==-1],hist=True,kde=True,bins=np.arange(minscore,maxscore+intervalscore,intervalscore),kde_kws={'linewidth': 3},label="false",norm_hist=True)
	plt.xlabel('Similarity Scores (higher is positive)')
	plt.ylabel('Density')
	plt.legend(loc="upper right")
	plt.title(title)
	plt.tight_layout()
	x = datetime.datetime.now().strftime("%c")
	title+=" "+x
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Find baseline')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/rdf_files/")
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/graphs/")
	parser.add_argument('-mp','--modelpath', metavar='model path',type=str,help='Model directory to load the model',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/models/claims-relatedness-model/claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27")
	parser.add_argument('-ep','--embedpath', metavar='embed path',type=str,help='Model directory to save and load embeddings',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/embeddings")
	parser.add_argument('-fcg','--fcgclass', metavar='FactCheckGraph class',type=str,choices=['co_occur','fred'])
	parser.add_argument('-gt','--graphtype', metavar='Graph Type Directed/Undirected',type=str,choices=['directed','undirected'],default='undirected')
	parser.add_argument('-bt','--baselinetype', metavar='Baseline Type All neighbors or K NearestNeighbors',type=str,choices=['all','knn'])
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	parser.add_argument('-n','--neighbors',metavar='Number of Neighbors',type=int,help='Number of Neighbors for KNN',default=0)
	args=parser.parse_args()
	# embed_claims(args.rdfpath,"roberta-base-nli-stsb-mean-tokens",args.embedpath,args.fcgtype)
	if args.baselinetype=='all':
		find_baseline(args.rdfpath,args.graphpath,args.modelpath,args.embedpath,args.graphtype,args.fcgclass,args.cpu)
	elif args.baselinetype=='knn':
		find_knn(args.rdfpath,args.graphpath,args.modelpath,args.embedpath,args.graphtype,args.fcgclass,args.cpu,args.neighbors)