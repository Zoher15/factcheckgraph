import argparse
import pandas as pd
import numpy as np 
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_adj_pairs(graph_path,fcg_class,kg_label):
	fcg_types={'fred':['tfcg','ffcg','ufcg'],'co-occur':['tfcg_co','ffcg_co','ufcg_co'],'backbone':['tfcg_bb','ffcg_bb','ufcg_bb']}
	fcg_labels=fcg_labels[fcg_class]
	fcg_path=os.path.join(graph_path,fcg_class)
	plt.figure()
	lw=2
	plt.plot([0, 1], [0, 1],color='navy',lw=lw,linestyle='--')
	title="true(adj) vs false(non-adj) pairs {} {}".format(kg_label,fcg_class)
	for fcg_label in fcg_labels:
		intersect_adj=pd.read_json(os.path.join(fcg_path,fcg_label,"intersect_adj_{}_{}_ID.json".format(kg_label,fcg_label)))
		intersect_nonadj=pd.read_json(os.path.join(fcg_path,fcg_label,"intersect_nonadj_{}_{}_ID.json".format(kg_label,fcg_label)))
		intersect_adj['label']=1
		intersect_adj['label']=0
		adj_nonadj=pd.concat([intersect_adj,intersect_nonadj],ignore_index=True)
		y=list(adj_nonadj['label'])
		scores=list(adj_nonadj['simil'])
		fpr,tpr,thresholds=metrics.roc_curve(y,scores, pos_label=1)
		plt.plot(fpr,tpr,lw=lw,label=fcg_label+' (%0.2f) '%metrics.auc(fpr,tpr))
	plt.set_xlabel('true positive rate')
	plt.set_xlabel('false positive rate')
	plt.legend(loc="upper right")
	plt.set_title(title)
	plt.tight_layout()
	plot_path=os.path.join(fcg_path,"plots")
	os.makedirs(plot_path,exist_ok=True)
	plt.savefig(os.path.join(plot_path,title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Plotting true(adjacent) pairs vs false (non-adjacent)')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	parser.add_argument('-fcg','--fcgclass', metavar='fcg class',type=str,choices=['fred','co-occur','backbone'],help='Class of FactCheckGraph to process')
	parser.add_argument('-kg','--kgtype', metavar='knowledgegraph type',type=str,choices=['dbpedia','wikidata'],help='DBPedia or Wikidata Graph')
	args=parser.parse_args()
	plot_adj_pairs(args.graphpath,args.fcgclass,args.kgtype)