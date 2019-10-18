import argparse
import pandas as pd
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_adj_pairs(graph_path,fcg_class,kg_label,corr_type):
	fcg_types={'fred':['tfcg','ffcg','ufcg'],'co-occur':['tfcg_co','ffcg_co','ufcg_co'],'backbone':['tfcg_bb','ffcg_bb','ufcg_bb']}
	fcg_labels=fcg_labels[fcg_class]
	fcg_path=os.path.join(graph_path,fcg_class)
	plt.figure()
	plt.set_xscale("log")
	title="pair correlation {} {}".format(kg_label,fcg_class)
	for fcg_label in fcg_labels:
		correlations=pd.read_csv(os.path.join(graph_path,fcg_class,fcg_label,"corr","corr_{}_{}.csv".format(kg_label,fcg_label)))
		plt.plot(correlations['index'],correlations[corr_type],label="{} - {}".format(kg_label,fcg_label))
	plt.set_xlabel("Number of pairs")
	plt.set_ylabel("Correlation")
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
	parser.add_argument('-c','--corr', metavar='correlation type',type=str,choices=['kendalltau','spearmanr','pearsonr'],help='Type of correlation')
	args=parser.parse_args()
	plot_adj_pairs(args.graphpath,args.fcgclass,args.kgtype,args.corr)