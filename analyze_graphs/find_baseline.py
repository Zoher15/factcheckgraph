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

def embed_claims(rdf_path,model_path,embed_path,claim_type):
	os.makedirs(embed_path, exist_ok=True)
	model = SentenceTransformer(model_path)
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
	claims_list=list(claims['claim_text'])
	claims_embeddings=model.encode(claims_list)
	with open(os.path.join(embed_path,claim_type+'_claims_embeddings_({}).tsv'.format(model_path.split("/")[-1])),'w',newline='') as f:
		for vector in claims_embeddings:
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)
	with open(os.path.join(embed_path,claim_type+'_claims_embeddings_labels_({}).tsv'.format(model_path.split("/")[-1])),'w',newline='') as f:
		vector=['claim_text','claimID','rating']
		tsv_output=csv.writer(f,delimiter='\t')
		tsv_output.writerow(vector)
		for i in range(len(claims)):
			vector=list((claims.iloc[i]['claim_text'],claims.iloc[i]['claimID'],claim_type))
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)

def find_baseline(rdf_path,model_path,embed_path,cpu):
	'''
	To have an accurate baseline for future: 1. remove claims that do no exist in the fred graph
	2. Remove claims that are near duplicates
	'''
	try:
		true_claims_embed=pd.read_csv(os.path.join(embed_path,"true_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	except FileNotFoundError:
		embed_claims(rdf_path,model_path,embed_path,claim_type)
		true_claims_embed=pd.read_csv(os.path.join(embed_path,"true_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	try:
		false_claims_embed=pd.read_csv(os.path.join(embed_path,"false_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	except FileNotFoundError:
		embed_claims(rdf_path,model_path,embed_path,claim_type)
		false_claims_embed=pd.read_csv(os.path.join(embed_path,"false_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	true_true=1-np.arccos(cosine_similarity(true_claims_embed,true_claims_embed))/np.pi
	true_false=1-np.arccos(cosine_similarity(true_claims_embed,false_claims_embed))/np.pi
	false_false=1-np.arccos(cosine_similarity(false_claims_embed,false_claims_embed))/np.pi
	np.fill_diagonal(true_true,np.nan)
	np.fill_diagonal(false_false,np.nan)

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
	true_scores=(true_true_mean*len(true_true)-true_false_true_mean*len(false_false))/(true_true_mean*len(true_true)+true_false_true_mean*len(false_false))
	false_scores=(true_false_false_mean*len(true_true)-false_false_mean*len(false_false))/(true_false_false_mean*len(true_true)+false_false_mean*len(false_false))
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
	plt.figure(figsize=(9, 8))
	sns.distplot(true_scores,hist=False,kde=True,kde_kws={'linewidth': 3},label="true")
	sns.distplot(false_scores,hist=False,kde=True,kde_kws={'linewidth': 3},label="false")
	plt.xlabel('Scores')
	plt.ylabel('Density')
	plt.legend(loc="upper right")
	plt.title(title)
	plt.tight_layout()
	x = datetime.datetime.now().strftime("%c")
	title+=" "+x
	plt.savefig(os.path.join(embed_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()


def find_knn(rdf_path,model_path,embed_path,cpu):
	'''
	To have an accurate baseline for future: 1. remove claims that do no exist in the fred graph
	2. Remove claims that are near duplicates
	'''
	try:
		true_claims_embed=pd.read_csv(os.path.join(embed_path,"true_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	except FileNotFoundError:
		embed_claims(rdf_path,model_path,embed_path,claim_type)
		true_claims_embed=pd.read_csv(os.path.join(embed_path,"true_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	try:
		false_claims_embed=pd.read_csv(os.path.join(embed_path,"false_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	except FileNotFoundError:
		embed_claims(rdf_path,model_path,embed_path,claim_type)
		false_claims_embed=pd.read_csv(os.path.join(embed_path,"false_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
	true_y=[1 for i in range(len(true_claims_embed))]
	false_y=[0 for i in range(len(false_claims_embed))]
	Y=np.array(true_y[:100]+false_y[:100])
	X=np.concatenate((true_claims_embed[:100],false_claims_embed[:100]),axis=0)
	n_neighbors = 5
	X=np.arccos(cosine_similarity(X,X))/np.pi
	np.fill_diagonal(X,np.nan)
	X=np.argpartition(X,n_neighbors)[:,:n_neighbors]
	Yh=np.apply_along_axis(lambda x:np.mean(Y[x]),1,X)
	plt.figure(figsize=(9, 8))
	lw=2
	plt.plot([0, 1], [0, 1],color='navy',lw=lw,linestyle='--')
	title="ROC KNN using angular similarity of claims K={}".format(n_neighbors)
	fpr,tpr,thresholds=metrics.roc_curve(Y,Yh, pos_label=1)
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
	title=title.replace('ROC','KDE')
	plt.figure(figsize=(9, 8))
	sns.distplot(Yh[Yh>=0.5],hist=False,kde=True,kde_kws={'linewidth': 3},label="true")
	sns.distplot(Yh[Yh<0.5],hist=False,kde=True,kde_kws={'linewidth': 3},label="false")
	plt.xlabel('Scores')
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
	parser.add_argument('-mp','--modelpath', metavar='model path',type=str,help='Model directory to load the model',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/models/claims-relatedness-model/claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27")
	parser.add_argument('-ep','--embedpath', metavar='embed path',type=str,help='Model directory to save and load embeddings',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/embeddings")
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	args=parser.parse_args()
	# embed_claims(args.rdfpath,"roberta-base-nli-stsb-mean-tokens",args.embedpath,args.fcgtype)
	# find_baseline(args.rdfpath,args.modelpath,args.embedpath,args.cpu)
	find_knn(args.rdfpath,args.modelpath,args.embedpath,args.cpu)