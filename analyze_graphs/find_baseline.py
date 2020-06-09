import os
import pandas as pd
import re
import numpy as np
import argparse
import rdflib
import codecs
import json
from sklearn import metrics
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
import sys
import datetime
import matplotlib
import matplotlib.pyplot as plt
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
	true_true=np.arccos(cosine_similarity(true_claims_embed,true_claims_embed))/np.pi
	true_false=np.arccos(cosine_similarity(true_claims_embed,false_claims_embed))/np.pi
	false_false=np.arccos(cosine_similarity(false_claims_embed,false_claims_embed))/np.pi
	np.fill_diagonal(true_true,np.nan)
	np.fill_diagonal(false_false,np.nan)

	plt.figure(figsize=(9, 8))
	lw=2
	plt.plot([0, 1], [0, 1],color='navy',lw=lw,linestyle='--')
	true_y=[0 for i in range(len(true_true))]
	false_y=[1 for i in range(len(false_false))]

	title="Baseline using angular distance of claims"

	true_true_mean=np.apply_along_axis(np.nanmean,1,true_true)
	true_false_true_mean=np.apply_along_axis(np.nanmean,1,true_false)
	false_false_mean=np.apply_along_axis(np.nanmean,1,false_false)
	true_false_false_mean=np.apply_along_axis(np.nanmean,0,true_false)

	fpr,tpr,thresholds=metrics.roc_curve(true_y+false_y,list(true_true_mean)+list(true_false_false_mean), pos_label=1)
	plt.plot(fpr,tpr,lw=lw,label='true claims mean (%0.2f) '%metrics.auc(fpr,tpr))

	fpr,tpr,thresholds=metrics.roc_curve(true_y+false_y,list(true_false_true_mean)+list(false_false_mean), pos_label=1)
	plt.plot(fpr,tpr,lw=lw,label='false claims mean (%0.2f) '%metrics.auc(fpr,tpr))

	true_true_min=np.apply_along_axis(np.nanmin,1,true_true)
	true_false_true_min=np.apply_along_axis(np.nanmin,1,true_false)
	false_false_min=np.apply_along_axis(np.nanmin,1,false_false)
	true_false_false_min=np.apply_along_axis(np.nanmin,0,true_false)

	fpr,tpr,thresholds=metrics.roc_curve(true_y+false_y,list(true_true_min)+list(true_false_false_min), pos_label=1)
	plt.plot(fpr,tpr,lw=lw,label='true claims min (%0.2f) '%metrics.auc(fpr,tpr))

	fpr,tpr,thresholds=metrics.roc_curve(true_y+false_y,list(true_false_true_min)+list(false_false_min), pos_label=1)
	plt.plot(fpr,tpr,lw=lw,label='false claims min (%0.2f) '%metrics.auc(fpr,tpr))

	true_true_max=np.apply_along_axis(np.nanmax,1,true_true)
	true_false_true_max=np.apply_along_axis(np.nanmax,1,true_false)
	false_false_max=np.apply_along_axis(np.nanmax,1,false_false)
	true_false_false_max=np.apply_along_axis(np.nanmax,0,true_false)

	fpr,tpr,thresholds=metrics.roc_curve(true_y+false_y,list(true_true_max)+list(true_false_false_max), pos_label=1)
	plt.plot(fpr,tpr,lw=lw,label='true claims max (%0.2f) '%metrics.auc(fpr,tpr))

	fpr,tpr,thresholds=metrics.roc_curve(true_y+false_y,list(true_false_true_max)+list(false_false_max), pos_label=1)
	plt.plot(fpr,tpr,lw=lw,label='false claims max (%0.2f) '%metrics.auc(fpr,tpr))

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

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Find baseline')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/rdf_files/")
	parser.add_argument('-mp','--modelpath', metavar='model path',type=str,help='Model directory to load the model',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/models/claims-relatedness-model/claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27")
	parser.add_argument('-ep','--embedpath', metavar='embed path',type=str,help='Model directory to save and load embeddings',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/embeddings")
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	args=parser.parse_args()
	# embed_claims(args.rdfpath,"roberta-base-nli-stsb-mean-tokens",args.embedpath,args.fcgtype)
	find_baseline(args.rdfpath,args.modelpath,args.embedpath,args.cpu)