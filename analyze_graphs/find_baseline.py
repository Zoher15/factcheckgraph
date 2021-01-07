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

def find_baseline(rdf_path,graph_path,model_path,embed_path,graph_type,fcg_class):
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
	#storing index of duplicate-ish claims to get rid of them from the data
	x_true=list(set([np.sort([i,j])[-1] for i in range(len(true_true)) for j in range(len(true_true)) if true_true[j][i]>0.825 and i!=j]))
	x_false=list(set([np.sort([i,j])[-1] for i in range(len(false_false)) for j in range(len(false_false)) if false_false[j][i]>0.825 and i!=j]))
	#deleting from the cross matrix
	true_true=np.delete(true_true,x_true,axis=0)
	true_true=np.delete(true_true,x_true,axis=1)
	true_false=np.delete(true_false,x_true,axis=0)
	true_false=np.delete(true_false,x_false,axis=1)
	false_false=np.delete(false_false,x_false,axis=0)
	false_false=np.delete(false_false,x_false,axis=1)
	####################################################################################################
	title="true vs true duplicates"
	plt.figure(figsize=(9, 8))
	sns.heatmap(true_true)
	plt.xlabel('True Claims')
	plt.ylabel('True Claims')
	plt.title(title)
	plt.tight_layout()
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+"_"+model_path.split("/")[-1]+".png"))
	plt.close()
	plt.clf()
	title="false vs false duplicates"
	plt.figure(figsize=(9, 8))
	sns.heatmap(false_false)
	plt.xlabel('False Claims')
	plt.ylabel('False Claims')
	plt.title(title)
	plt.tight_layout()
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+"_"+model_path.split("/")[-1]+".png"))
	plt.close()
	plt.clf()
	title="true vs false duplicates"
	plt.figure(figsize=(9, 8))
	sns.heatmap(true_false)
	plt.xlabel('False Claims')
	plt.ylabel('True Claims')
	plt.title(title)
	plt.tight_layout()
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+"_"+model_path.split("/")[-1]+".png"))
	plt.close()
	plt.clf()
	####################################################################################################
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
	intervalscore=float(maxscore-minscore)/100
	print(intervalscore)
	print(minscore)
	print(maxscore)
	sns.distplot(true_scores,hist=True,kde=False,bins=np.arange(minscore,maxscore+intervalscore,intervalscore),kde_kws={'linewidth': 3},label="true",norm_hist=True)
	sns.distplot(false_scores,hist=True,kde=False,bins=np.arange(minscore,maxscore+intervalscore,intervalscore),kde_kws={'linewidth': 3},label="false",norm_hist=True)
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

def find_knn(k,rdf_path,graph_path,model_path,embed_path,graph_type,fcg_class,n_neighbors):
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
	X_=np.concatenate((true_claims_embed,false_claims_embed),axis=0)
	#find pairwise cosine similarity
	X=np.arccos(np.clip(cosine_similarity(X_,X_),-1,1))/np.pi
	X_t=np.arccos(np.clip(cosine_similarity(X_,true_claims_embed),-1,1))/np.pi
	# data from claim labels: text, rating, claimID
	Xlabels=np.concatenate((true_claims_labels[1:],false_claims_labels[1:]),axis=0)
	Xlabels_t=true_claims_labels[1:]
	np.fill_diagonal(X,np.nan)
	np.fill_diagonal(X_t[:len(true_y),:],np.nan)
	########################################################################################### de duplication
	#storing index of duplicate-ish claims to get rid of them from the data
	x_true=list(set([np.sort([i,j])[-1] for i in range(len(true_claims_embed)) for j in range(len(true_claims_embed)) if X[j][i]<=0.175 and i!=j]))
	x_false=list(set([np.sort([i,j])[-1] for i in range(len(true_claims_embed),len(false_claims_embed)+len(true_claims_embed)) for j in range(len(true_claims_embed),len(false_claims_embed)+len(true_claims_embed)) if X[j][i]<=0.175 and i!=j]))# to put the index in right spot
	#deleting from the cross matrix
	X=np.delete(X,x_true+x_false,axis=0)
	X=np.delete(X,x_true+x_false,axis=1)
	X_t=np.delete(X_t,x_true+x_false,axis=0)
	X_t=np.delete(X_t,x_true,axis=1)
	Xlabels=np.delete(Xlabels,x_true+x_false,axis=0)
	Xlabels_t=np.delete(Xlabels_t,x_true,axis=0)
	Y=np.delete(Y,x_true+x_false,axis=0)
	####################################################################################################
	#Finding n_neighbors
	Xn=np.argpartition(X,n_neighbors)[:,:n_neighbors]
	Xn_t=np.argpartition(X_t,n_neighbors)[:,:n_neighbors]
	# Yh=np.apply_along_axis(lambda x:np.mean(Y[x]),1,Xn)
	#Intializing Yh to 0
	Yh=np.array([float(0) for i in range(Xn.shape[0])])
	Yh_t=np.array([float(0) for i in range(Xn_t.shape[0])])
	#text in Xlabels
	Xlabel=np.apply_along_axis(lambda x:Xlabels[x,0],1,Xn)
	################################################################################################################
	###############################"_t" indicates only true claims are being used to check the claims
	Xlabel_t=np.apply_along_axis(lambda x:Xlabels_t[x,0],1,Xn_t)
	#claimID in Xlabels
	XclaimID=np.apply_along_axis(lambda x:Xlabels[x,1],1,Xn)
	XclaimID_t=np.apply_along_axis(lambda x:Xlabels_t[x,1],1,Xn_t)
	#rating in Xlabels
	Xrating=np.apply_along_axis(lambda x:Xlabels[x,2],1,Xn)
	Xrating_t=np.apply_along_axis(lambda x:Xlabels_t[x,2],1,Xn_t)
	Xndict={}
	Xndict_t={}
	for i in range(Xn.shape[0]):
		text=chunkstring(Xlabels[i][0],100)
		text_t=chunkstring(Xlabels[i][0],100)
		#Xlabels[i][1] gives the claimID. We want to use the claimID as the key
		claimID=Xlabels[i][1]
		Xndict[claimID]={}
		Xndict_t[claimID]={}
		aggregate=0
		aggregate_t=0
		Xndict[claimID]['neighbors']={}
		Xndict_t[claimID]['neighbors']={}
		for j in range(Xn.shape[1]):
			Xndict[claimID]['neighbors'][int(Xn[i,j])]={}
			Xndict_t[claimID]['neighbors'][int(Xn_t[i,j])]={}
			#
			n_text=chunkstring(Xlabels[Xn[i,j]][0],100)
			n_text_t=chunkstring(Xlabels_t[Xn_t[i,j]][0],100)
			#
			Xndict[claimID]['neighbors'][Xn[i,j]]={'n_text':n_text,'claimID':XclaimID[i,j],'rating':Xrating[i,j],'dist':round(X[i,Xn[i,j]],2)}
			Xndict_t[claimID]['neighbors'][Xn_t[i,j]]={'n_text':n_text_t,'claimID':XclaimID_t[i,j],'rating':Xrating_t[i,j],'dist':round(X_t[i,Xn_t[i,j]],2)}
			#
			aggregate+=Y[Xn[i,j]]*(1-X[i,Xn[i,j]])
			aggregate_t+=Y[Xn_t[i,j]]*(1-X_t[i,Xn_t[i,j]])
		#
		Yh[i]=float(aggregate)/n_neighbors
		Yh_t[i]=float(aggregate_t)/n_neighbors
		#
		Xndict[claimID]['text']=text
		Xndict_t[claimID]['text']=text_t
		#
		Xndict[claimID]['predscore']=round(Yh[i],2)
		Xndict_t[claimID]['predscore']=round(Yh_t[i],2)
		#
		Xndict[claimID]['score']=float(Y[i])
		Xndict_t[claimID]['score']=float(Y[i])
		#
		Xndict[claimID]['neighbors']=OrderedDict(sorted(Xndict[claimID]['neighbors'].items(), key=lambda t: t[1]['dist']))
		Xndict_t[claimID]['neighbors']=OrderedDict(sorted(Xndict_t[claimID]['neighbors'].items(), key=lambda t: t[1]['dist']))
		#
		Xndict[claimID]=OrderedDict(sorted(Xndict[claimID].items(), key=lambda t: t[0],reverse=True))
		Xndict_t[claimID]=OrderedDict(sorted(Xndict_t[claimID].items(), key=lambda t: t[0],reverse=True))

	#True claims (label=1)
	Xndict_1={t[0]:t[1] for t in Xndict.items() if t[1]['score']==1}
	Xndict_1_t={t[0]:t[1] for t in Xndict_t.items() if t[1]['score']==1}
	#False claims (label=-1)
	Xndict_0={t[0]:t[1] for t in Xndict.items() if t[1]['score']==-1}
	Xndict_0_t={t[0]:t[1] for t in Xndict_t.items() if t[1]['score']==-1}
	#False claims with high predscore
	Xndict_1=OrderedDict(sorted(Xndict_1.items(), key=lambda t:t[1]['predscore'], reverse=True))
	Xndict_1_t=OrderedDict(sorted(Xndict_1_t.items(), key=lambda t:t[1]['predscore'], reverse=True))
	#
	Xndict_0=OrderedDict(sorted(Xndict_0.items(), key=lambda t:t[1]['predscore'], reverse=True))
	Xndict_0_t=OrderedDict(sorted(Xndict_0_t.items(), key=lambda t:t[1]['predscore'], reverse=True))
	#
	with codecs.open(os.path.join(embed_path,"knn_1_{}.json".format(n_neighbors)),"w","utf-8") as f:
		f.write(json.dumps(Xndict_1,indent=6,ensure_ascii=False))
	with codecs.open(os.path.join(embed_path,"knn_1_{}_t.json".format(n_neighbors)),"w","utf-8") as f:
		f.write(json.dumps(Xndict_1_t,indent=6,ensure_ascii=False))
	#
	with codecs.open(os.path.join(embed_path,"knn_0_{}.json".format(n_neighbors)),"w","utf-8") as f:
		f.write(json.dumps(Xndict_0,indent=6,ensure_ascii=False))
	with codecs.open(os.path.join(embed_path,"knn_0_{}_t.json".format(n_neighbors)),"w","utf-8") as f:
		f.write(json.dumps(Xndict_0_t,indent=6,ensure_ascii=False))
	#
	Xndict_0_high={t[0]:t[1] for t in Xndict.items() if t[1]['score']==-1 and t[1]['predscore']>0}
	Xndict_0_high_t={t[0]:t[1] for t in Xndict_t.items() if t[1]['score']==-1 and t[1]['predscore']>0}
	#
	Xndict_0_high=OrderedDict(sorted(Xndict_0_high.items(), key=lambda t:t[1]['predscore'],reverse=True))
	Xndict_0_high_t=OrderedDict(sorted(Xndict_0_high_t.items(), key=lambda t:t[1]['predscore'],reverse=True))
	#
	with codecs.open(os.path.join(embed_path,"knn_0_high_{}.json".format(n_neighbors)),"w","utf-8") as f:
		f.write(json.dumps(Xndict_0_high,indent=6,ensure_ascii=False))
	with codecs.open(os.path.join(embed_path,"knn_0_high_{}_t.json".format(n_neighbors)),"w","utf-8") as f:
		f.write(json.dumps(Xndict_0_high_t,indent=6,ensure_ascii=False))
	lw=2
	ogtitle="KNN using angular similarity of claims K={}".format(n_neighbors)
	##################################################################################ROC PLOT
	plt.figure(figsize=(9, 8))
	plt.plot([0, 1], [0, 1],color='navy',lw=lw,linestyle='--')
	title="ROC "+ogtitle
	fpr,tpr,thresholds=metrics.roc_curve(Y,Yh,pos_label=1)
	auc=metrics.roc_auc_score(Y,Yh)
	plt.plot(fpr,tpr,lw=lw,label='AUC (%0.2f) '%auc)
	plt.xlabel('True Positive Rate')
	plt.ylabel('False Positive Rate')
	plt.legend(loc="lower right")
	plt.title(title)
	plt.tight_layout()
	title+=" "+datetime.datetime.now().strftime("%c")
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()
	##################################################################################PR PLOT
	title="PR "+ogtitle
	plt.figure(figsize=(9, 8))
	precision,recall,thresholds=metrics.precision_recall_curve(Y,Yh,pos_label=1)
	avg_pr=metrics.average_precision_score(Y,Yh)
	plt.plot(recall,precision,lw=lw,label='AVG_Pr (%0.2f) '%avg_pr)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc="upper right")
	plt.title(title)
	plt.tight_layout()
	title+=" "+datetime.datetime.now().strftime("%c")
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()
	##################################################################################F1 PLOT
	title="F1 "+ogtitle
	plt.figure(figsize=(9, 8))
	f1scores=[2*(precision[i]*recall[i])/(precision[i]+recall[i]) for i in range(len(thresholds))]
	f1=max(f1scores)
	plt.plot(thresholds,f1scores,lw=lw,label='Max F1 (%0.2f) '%f1)
	plt.xlabel('Thresholds')
	plt.ylabel('F1 Scores')
	plt.legend(loc="upper right")
	plt.title(title)
	plt.tight_layout()
	title+=" "+datetime.datetime.now().strftime("%c")
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()
	##################################################################################KDE PLOT
	title="KDE "+ogtitle
	plt.figure(figsize=(9, 8))
	minscore=np.min(list(Yh[Y==1])+list(Yh[Y==-1]))
	maxscore=np.max(list(Yh[Y==1])+list(Yh[Y==-1]))
	intervalscore=float(maxscore-minscore)/100
	print(intervalscore)
	print(minscore)
	print(maxscore)
	sns.distplot(Yh[Y==1],hist=True,kde=False,bins=np.arange(minscore,maxscore+intervalscore,intervalscore),kde_kws={'linewidth': 3},label="true",norm_hist=True)
	sns.distplot(Yh[Y==-1],hist=True,kde=False,bins=np.arange(minscore,maxscore+intervalscore,intervalscore),kde_kws={'linewidth': 3},label="false",norm_hist=True)
	plt.xlabel('Similarity Scores (higher is positive)')
	plt.ylabel('Density')
	plt.legend(loc="upper right")
	plt.title(title)
	plt.tight_layout()
	title+=" "+datetime.datetime.now().strftime("%c")
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()
	##################################################################################################################################
	ogtitle="KNN using angular similarity of claims K={} t".format(n_neighbors)
	##################################################################################ROC PLOT
	plt.figure(figsize=(9, 8))
	plt.plot([0, 1], [0, 1],color='navy',lw=lw,linestyle='--')
	title="ROC "+ogtitle
	fpr,tpr,thresholds=metrics.roc_curve(Y,Yh_t,pos_label=1)
	auc_t=metrics.roc_auc_score(Y,Yh_t)
	plt.plot(fpr,tpr,lw=lw,label='AUC (%0.2f) '%auc_t)
	plt.xlabel('True Positive Rate')
	plt.ylabel('False Positive Rate')
	plt.legend(loc="lower right")
	plt.title(title)
	plt.tight_layout()
	title+=" "+datetime.datetime.now().strftime("%c")
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()
	##################################################################################PR PLOT
	title="PR "+ogtitle
	plt.figure(figsize=(9, 8))
	precision,recall,thresholds=metrics.precision_recall_curve(Y,Yh_t,pos_label=1)
	avg_pr_t=metrics.average_precision_score(Y,Yh_t)
	plt.plot(recall,precision,lw=lw,label='AVG_Pr (%0.2f) '%avg_pr_t)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc="upper right")
	plt.title(title)
	plt.tight_layout()
	title+=" "+datetime.datetime.now().strftime("%c")
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()
	##################################################################################F1 PLOT
	title="F1 "+ogtitle
	plt.figure(figsize=(9, 8))
	f1scores=[2*(precision[i]*recall[i])/(precision[i]+recall[i]) for i in range(len(thresholds))]
	f1_t=max(f1scores)
	plt.plot(thresholds,f1scores,lw=lw,label='Max F1 (%0.2f) '%f1_t)
	plt.xlabel('Thresholds')
	plt.ylabel('F1 Scores')
	plt.legend(loc="upper right")
	plt.title(title)
	plt.tight_layout()
	title+=" "+datetime.datetime.now().strftime("%c")
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()
	##################################################################################KDE PLOT
	title="KDE "+ogtitle
	plt.figure(figsize=(9, 8))
	minscore=np.min(list(Yh_t[Y==1])+list(Yh_t[Y==-1]))
	maxscore=np.max(list(Yh_t[Y==1])+list(Yh_t[Y==-1]))
	intervalscore=float(maxscore-minscore)/20
	print(intervalscore)
	print(minscore)
	print(maxscore)
	sns.distplot(Yh_t[Y==1],hist=True,kde=True,bins=np.arange(minscore,maxscore+intervalscore,intervalscore),kde_kws={'linewidth': 3},label="true",norm_hist=True)
	sns.distplot(Yh_t[Y==-1],hist=True,kde=True,bins=np.arange(minscore,maxscore+intervalscore,intervalscore),kde_kws={'linewidth': 3},label="false",norm_hist=True)
	plt.xlabel('Similarity Scores (higher is positive)')
	plt.ylabel('Density')
	plt.legend(loc="upper right")
	plt.title(title)
	plt.tight_layout()
	title+=" "+datetime.datetime.now().strftime("%c")
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()
	return k,auc,auc_t,avg_pr,avg_pr_t,f1,f1_t

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Find baseline')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/rdf_files/")
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/graphs/")
	parser.add_argument('-mp','--modelpath', metavar='model path',type=str,help='Model directory to load the model',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/models/claims-relatedness-model/claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27")
	parser.add_argument('-ep','--embedpath', metavar='embed path',type=str,help='Model directory to save and load embeddings',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/embeddings")
	parser.add_argument('-fcg','--fcgclass', metavar='FactCheckGraph class',type=str,choices=['co_occur','fred'])
	parser.add_argument('-gt','--graphtype', metavar='Graph Type Directed/Undirected',type=str,choices=['directed','undirected'],default='undirected')
	parser.add_argument('-bt','--baselinetype', metavar='Baseline Type All neighbors or K NearestNeighbors',type=str,choices=['all','knn'])
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available')
	# parser.add_argument('-n','--neighbors',metavar='Number of Neighbors',type=int,help='Number of Neighbors for KNN',default=0)
	args=parser.parse_args()
	# embed_claims(args.rdfpath,"roberta-base-nli-stsb-mean-tokens",args.embedpath,args.fcgtype)
	if args.baselinetype=='all':
		find_baseline(args.rdfpath,args.graphpath,args.modelpath,args.embedpath,args.graphtype,args.fcgclass)
	elif args.baselinetype=='knn':
		rocs=[]
		plotrange=[i for i in range(0,41,10)]
		plotrange[0]+=1
		if args.cpu:
			pool=mp.Pool(processes=args.cpu)
			results=[pool.apply_async(find_knn, args=(k,args.rdfpath,args.graphpath,args.modelpath,args.embedpath,args.graphtype,args.fcgclass,k)) for k in plotrange]
			results=sorted([p.get() for p in results],key=lambda t:t[0])
		else:
			rocs=[]
			for k in plotrange:
				results.append(find_knn(k,args.rdfpath,args.graphpath,args.modelpath,args.embedpath,args.graphtype,args.fcgclass,k))
		aucs=[t[1] for t in results]
		aucs_t=[t[2] for t in results]
		avg_pr=[t[3] for t in results]
		avg_pr_t=[t[4] for t in results]
		f1=[t[5] for t in results]
		f1_t=[t[6] for t in results]
		plt.figure(figsize=(9, 8))
		plt.plot(plotrange,aucs,'g^',markersize=7,label='true+false')
		plt.plot(plotrange,aucs_t,'bs',markersize=7,label='true')
		plt.legend(loc="center right")
		plt.xlabel('number of neighbors')
		plt.ylabel('roc auc')
		x = datetime.datetime.now().strftime("%c")
		plt.savefig(os.path.join(args.embedpath,"roc_auc_{}.png".format(x)))
		plt.close()
		plt.clf()
		plt.figure(figsize=(9, 8))
		plt.plot(plotrange,avg_pr,'g^',markersize=7,label='true+false')
		plt.plot(plotrange,avg_pr_t,'bs',markersize=7,label='true')
		plt.legend(loc="center right")
		plt.xlabel('number of neighbors')
		plt.ylabel('average precision score')
		x = datetime.datetime.now().strftime("%c")
		plt.savefig(os.path.join(args.embedpath,"avg_pr_{}.png".format(x)))
		plt.close()
		plt.clf()
		plt.figure(figsize=(9, 8))
		plt.plot(plotrange,f1,'g^',markersize=7,label='true+false')
		plt.plot(plotrange,f1_t,'bs',markersize=7,label='true')
		plt.legend(loc="center right")
		plt.xlabel('number of neighbors')
		plt.ylabel('max f1 score')
		x = datetime.datetime.now().strftime("%c")
		plt.savefig(os.path.join(args.embedpath,"f1_{}.png".format(x)))
		plt.close()
		plt.clf()

