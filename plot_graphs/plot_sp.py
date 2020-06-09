import argparse
import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn import metrics
import json
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import datetime
import codecs
from statistics import mean

def aggregate_edge_data(evalues,mode):
	#mode can be dist or weight
	edgepair_weights=[]
	for edgepair,e2values in evalues.items():
		edgepair_weights.append(e2values[mode])
	return sum(edgepair_weights)

def aggregate_weights(claim_D,mode,mode2):
	#mode can be max, min, sum, mean
	#mode2 can be w or d
	edge_weights=[]
	for edge,evalues in claim_D.items():
		if type(evalues)!=list:
			#edge looks like this: "('db:John_McCain', 'db:United_States_Senate', 0.09, 1.25)"
			#u is the source node,v is the target node, w the special weight, d the 1/similarity
			u,v,w,d=eval(edge.replace("inf","np.inf"))
			edge_weights.append(eval(mode2))
	return eval("{}(edge_weights)".format(mode))

def plot_roc(graph_path,fcg_class):
	tfcg_types={"co_occur":"tfcg_co","fred":"tfcg"}
	embed={'roberta-base-nli-stsb-mean-tokens':'e1','claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27':'e2'}
	mode={'d':'d1','w':'d2'}
	aggmode={'mean':'a2','max':'a3','min':'a4'}
	plt.figure(figsize=(9, 8))
	lw=2
	plt.plot([0, 1], [0, 1],color='navy',lw=lw,linestyle='--')
	title="{} shortest path setence embedded paths".format(fcg_class)
	read_path=os.path.join(graph_path,fcg_class,"paths",tfcg_types[fcg_class])
	plot_path=os.path.join(graph_path,fcg_class,"plots")
	for e in list(embed.keys()):
		for d in list(mode.keys()):
			for a in list(aggmode.keys()):
				with codecs.open(os.path.join(read_path+"_true_({})".format(e),"paths_{}.json".format(d)),"r","utf-8") as f: 
					true_paths=json.loads(f.read())
				with codecs.open(os.path.join(read_path+"_false_({})".format(e),"paths_{}.json".format(d)),"r","utf-8") as f: 
					false_paths=json.loads(f.read())
				true_scores=list(map(lambda t:aggregate_weights(t[1],a,d),true_paths.items()))
				false_scores=list(map(lambda t:aggregate_weights(t[1],a,d),false_paths.items()))
				true_y=[0 for i in range(len(true_scores))]
				false_y=[1 for i in range(len(false_scores))]
				y=true_y+false_y
				scores=true_scores+false_scores
				fpr,tpr,thresholds=metrics.roc_curve(y,scores, pos_label=1)
				plt.plot(fpr,tpr,lw=lw,label=embed[e]+mode[d]+aggmode[a]+' (%0.2f) '%metrics.auc(fpr,tpr))
	#plotting the kg upperline ROC
	# kg_adj=pd.read_json(os.path.join(graphpath,"kg",kg_label,fcg_class,"intersect_adj_{}_{}_ID.json".format(fcg_class,kg_label)))
	# kg_nonadj=pd.read_json(os.path.join(graphpath,"kg",kg_label,fcg_class,"intersect_nonadj_{}_{}_ID.json".format(fcg_class,kg_label)))
	# kg_adj['label']=1
	# kg_nonadj['label']=0
	# adj_nonadj=pd.concat([kg_adj,kg_nonadj],ignore_index=True)
	# y=list(adj_nonadj['label'])
	# scores=list(adj_nonadj['simil'])
	# fpr,tpr,thresholds=metrics.roc_curve(y,scores, pos_label=1)
	# plt.plot(fpr,tpr,lw=lw,label=kg_label+' (%0.2f) '%metrics.auc(fpr,tpr))
	#Setting figure parameters
	# plt.xlabel('true positive rate')
	# plt.ylabel('false positive rate')
	plt.xlabel('True Positive Rate')
	plt.ylabel('False Positive Rate')
	plt.legend(loc="lower right")
	plt.title(title)
	plt.tight_layout()
	os.makedirs(plot_path,exist_ok=True)
	x = datetime.datetime.now().strftime("%c")
	title+=" "+x
	plt.savefig(os.path.join(plot_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()

def plot_dist(graph_path,fcg_class):
	tfcg_types={"co_occur":"tfcg_co","fred":"tfcg"}
	embed={'roberta-base-nli-stsb-mean-tokens':'e1','claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27':'e2'}
	mode={'d':'d1','w':'d2'}
	aggmode={'mean':'a2','max':'a3','min':'a4'}
	read_path=os.path.join(graph_path,fcg_class,"paths",tfcg_types[fcg_class])
	plot_path=os.path.join(graph_path,fcg_class,"plots")
	for e in list(embed.keys()):
		for d in list(mode.keys()):
			for a in list(aggmode.keys()):
				label=embed[e]+mode[d]+aggmode[a]
				plt.figure(figsize=(9, 8))
				title="{} shortest path setence embedded paths".format(fcg_class+"_"+label)
				with codecs.open(os.path.join(read_path+"_true_({})".format(e),"paths_{}.json".format(d)),"r","utf-8") as f: 
					true_paths=json.loads(f.read())
				with codecs.open(os.path.join(read_path+"_false_({})".format(e),"paths_{}.json".format(d)),"r","utf-8") as f: 
					false_paths=json.loads(f.read())
				true_scores=list(map(lambda t:aggregate_weights(t[1],a,d),true_paths.items()))
				false_scores=list(map(lambda t:aggregate_weights(t[1],a,d),false_paths.items()))
				sns.distplot(true_scores,hist=False,kde=True,kde_kws={'linewidth': 3},label="true_"+label)
				sns.distplot(false_scores,hist=False,kde=True,kde_kws={'linewidth': 3},label="false_"+label)
				plt.xlabel('Scores')
				plt.ylabel('Density')
				plt.legend(loc="upper right")
				plt.title(title)
				plt.tight_layout()
				os.makedirs(plot_path,exist_ok=True)
				x = datetime.datetime.now().strftime("%c")
				title+=" "+x
				plt.savefig(os.path.join(plot_path,title.replace(" ","_")+".png"))
				plt.close()
				plt.clf()

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Plotting true(adjacent) pairs vs false (non-adjacent)')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-fcg','--fcgclass', metavar='fcg class',type=str,help='Class of FactCheckGraph to process')
	parser.add_argument('-pt','--plottype', metavar='plot type',type=str,choices=['roc','dist'],help='Class of graph to plot')
	args=parser.parse_args()
	if args.plottype=='roc':
		plot_roc(args.graphpath,args.fcgclass)
	elif args.plottype=='dist':
		plot_dist(args.graphpath,args.fcgclass)