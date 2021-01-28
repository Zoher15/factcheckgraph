import argparse
import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn import metrics
import json
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 8]
plt.rcParams['figure.dpi'] = 100
matplotlib.use('Agg')
import os
import datetime
import codecs

def plot_roc(graph_path,fcg_class,fcg_type,graph_type,prefix):
	fcg_types={"co_occur":{"tfcg":"tfcg_co","ffcg":"ffcg_co"},"fred":{"tfcg":"tfcg","ffcg":"ffcg","ufcg":"ufcg"}}
	embeds={'roberta-base-nli-stsb-mean-tokens':'e1'}#,'claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27':'e2'}
	dists={'w':'d1','d':'d2'}#,'f':'d3'}
	aggs={'mean':'a1','max':'a2'}#,'median':'a3'}#,'domb':'a4'}
	if fcg_type:
		true_read_path=os.path.join(graph_path,fcg_class,"paths","true_"+fcg_types[fcg_class][fcg_type])
		false_read_path=os.path.join(graph_path,fcg_class,"paths","false_"+fcg_types[fcg_class][fcg_type])
	else:
		read_path=os.path.join(graph_path,fcg_class,"paths")	
	plot_path=os.path.join(graph_path,fcg_class,"plots")
	plot_dict={}
	plot_dict['roc']={}
	plot_dict['pr']={}
	plot_dict['f1']={}
	for embed in list(embeds.keys()):
		for dist in list(dists.keys()):
			for agg in list(aggs.keys()):
				if fcg_type:
					with codecs.open(os.path.join(true_read_path+"_({})".format(embed),"paths_"+graph_type+"_"+dist+"_"+agg+".json"),"r","utf-8") as f: 
						true_paths=json.loads(f.read())
					with codecs.open(os.path.join(false_read_path+"_({})".format(embed),"paths_"+graph_type+"_"+dist+"_"+agg+".json"),"r","utf-8") as f: 
						false_paths=json.loads(f.read())
				else:
					with codecs.open(os.path.join(read_path,"true_scores_"+dist+"_({})".format(embed)+"_"+agg+".json"),"r","utf-8") as f: 
						true_paths=json.loads(f.read())
					with codecs.open(os.path.join(read_path,"false_scores_"+dist+"_({})".format(embed)+"_"+agg+".json"),"r","utf-8") as f: 
						false_paths=json.loads(f.read())
				###################################################################################################
				true_scores=[eval(t[0])[1] for t in true_paths.items()]
				false_scores=[eval(t[0])[1] for t in false_paths.items()]
				true_y=[1 for i in range(len(true_scores))]
				false_y=[0 for i in range(len(false_scores))]
				y=true_y+false_y
				scores=true_scores+false_scores
				fpr,tpr,thresholds=metrics.roc_curve(y,scores, pos_label=1)
				precision,recall,thresholds=metrics.precision_recall_curve(y,scores, pos_label=1)
				f1scores=[2*(precision[i]*recall[i])/(precision[i]+recall[i]) for i in range(len(thresholds))]
				mlabel=embeds[embed]+dists[dist]+aggs[agg]
				plot_dict['roc'][mlabel]={}
				plot_dict['pr'][mlabel]={}
				plot_dict['f1'][mlabel]={}
				plot_dict['roc'][mlabel]['fpr']=fpr
				plot_dict['roc'][mlabel]['tpr']=tpr
				plot_dict['roc'][mlabel]['label']=mlabel+' AUC (%0.2f) '%metrics.roc_auc_score(y,scores)
				plot_dict['pr'][mlabel]['precision']=precision
				plot_dict['pr'][mlabel]['recall']=recall
				plot_dict['pr'][mlabel]['label']=mlabel+' AVG_Pr (%0.2f) '%metrics.average_precision_score(y,scores)
				plot_dict['f1'][mlabel]['thresholds']=thresholds
				plot_dict['f1'][mlabel]['f1scores']=f1scores
				plot_dict['f1'][mlabel]['label']=mlabel+' MAX_F1 (%0.2f) '%max(f1scores)
	#Plotting
	lw=2
	title=prefix+" "+fcg_type+" "+fcg_class+" shortest path"
	#Plot ROC
	plt.figure()
	roctitle="ROC "+title
	plt.plot([0, 1], [0, 1],color='navy',lw=lw,linestyle='--')
	for k in plot_dict['roc'].keys():
		plt.plot(plot_dict['roc'][k]['fpr'],plot_dict['roc'][k]['tpr'],lw=lw,label=plot_dict['roc'][k]['label'])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.title(roctitle)
	plt.tight_layout()
	os.makedirs(plot_path,exist_ok=True)
	roctitle+=" "+datetime.datetime.now().strftime("%c")
	plt.savefig(os.path.join(plot_path,roctitle.replace(" ","_")+".png"))
	plt.close()
	plt.clf()
	#Plot Precision Recall
	plt.figure()
	prtitle="PR "+title
	for k in plot_dict['pr'].keys():
		plt.plot(plot_dict['pr'][k]['recall'],plot_dict['pr'][k]['precision'],lw=lw,label=plot_dict['pr'][k]['label'])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc="upper right")
	plt.title(prtitle)
	plt.tight_layout()
	os.makedirs(plot_path,exist_ok=True)
	prtitle+=" "+datetime.datetime.now().strftime("%c")
	plt.savefig(os.path.join(plot_path,prtitle.replace(" ","_")+".png"))
	plt.close()
	plt.clf()
	#F1 Scores Thresholds
	plt.figure()
	f1title="F1 "+title
	for k in plot_dict['f1'].keys():
		plt.plot(plot_dict['f1'][k]['thresholds'],plot_dict['f1'][k]['f1scores'],lw=lw,label=plot_dict['f1'][k]['label'])
	plt.xlabel('Thresholds')
	plt.ylabel('F1 Scores')
	plt.legend(loc="upper right")
	plt.title(f1title)
	plt.tight_layout()
	os.makedirs(plot_path,exist_ok=True)
	f1title+=" "+datetime.datetime.now().strftime("%c")
	plt.savefig(os.path.join(plot_path,f1title.replace(" ","_")+".png"))
	plt.close()
	plt.clf()

def plot_dist(graph_path,fcg_class,fcg_type,graph_type,prefix):
	fcg_types={"co_occur":{"tfcg":"tfcg_co","ffcg":"ffcg_co"},"fred":{"tfcg":"tfcg","ffcg":"ffcg","ufcg":"ufcg"}}
	embeds={'roberta-base-nli-stsb-mean-tokens':'e1'}#,'claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27':'e2'}
	dists={'w':'d1','d':'d2'}#,'f':'d3'}
	aggs={'mean':'a1','max':'a2'}#,'median':'a3'}#,'domb':'a4'}
	if fcg_type:
		true_read_path=os.path.join(graph_path,fcg_class,"paths","true_"+fcg_types[fcg_class][fcg_type])
		false_read_path=os.path.join(graph_path,fcg_class,"paths","false_"+fcg_types[fcg_class][fcg_type])
	else:
		read_path=os.path.join(graph_path,fcg_class,"paths")	
	plot_path=os.path.join(graph_path,fcg_class,"plots")
	for embed in list(embeds.keys()):
		for dist in list(dists.keys()):
			for agg in list(aggs.keys()):
				label=embeds[embed]+dists[dist]+aggs[agg]
				plt.figure()
				if fcg_type:
					title=prefix+" histogram "+fcg_type+"_"+fcg_class+"_"+label+" shortest path"
					with codecs.open(os.path.join(true_read_path+"_({})".format(embed),"paths_"+graph_type+"_"+dist+"_"+agg+".json"),"r","utf-8") as f: 
						true_paths=json.loads(f.read())
					with codecs.open(os.path.join(false_read_path+"_({})".format(embed),"paths_"+graph_type+"_"+dist+"_"+agg+".json"),"r","utf-8") as f: 
						false_paths=json.loads(f.read())
				else:
					title=prefix+" histogram "+fcg_class+"_"+label+" shortest path"
					with codecs.open(os.path.join(read_path,"true_scores_"+dist+"_({})".format(embed)+"_"+agg+".json"),"r","utf-8") as f: 
						true_paths=json.loads(f.read())
					with codecs.open(os.path.join(read_path,"false_scores_"+dist+"_({})".format(embed)+"_"+agg+".json"),"r","utf-8") as f: 
						false_paths=json.loads(f.read())
				###################################################################################################
				true_scores=[eval(t[0])[1] for t in true_paths.items()]
				false_scores=[eval(t[0])[1] for t in false_paths.items()]
				minscore=np.floor(min(true_scores+false_scores))
				maxscore=np.ceil(max(true_scores+false_scores))
				intervalscore=float(maxscore-minscore)/100
				print(intervalscore)
				print(minscore)
				print(maxscore)
				try:
					plotrange=np.arange(minscore,maxscore+intervalscore,intervalscore)
					sns.distplot(true_scores,hist=True,kde=False,bins=plotrange,kde_kws={'linewidth': 3},label="true_"+label)
					sns.distplot(false_scores,hist=True,kde=False,bins=plotrange,kde_kws={'linewidth': 3},label="false_"+label)
				except ValueError:
					sns.distplot(true_scores,hist=True,kde=False,kde_kws={'linewidth': 3},label="true_"+label)
					sns.distplot(false_scores,hist=True,kde=False,kde_kws={'linewidth': 3},label="false_"+label)
				plt.xlabel('Proximity Scores (higher is positive)')
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
	parser.add_argument('-ft','--fcgtype', metavar='FCG Type',type=str,choices=['tfcg','ffcg','tfcg_co','ffcg_co','ufcg'])
	parser.add_argument('-pt','--plottype', metavar='plot type',type=str,choices=['roc','dist'],help='Class of graph to plot')
	parser.add_argument('-pr','--prefix', metavar='prefix for plot title',type=str,help='Add prefix to plot title',default="")
	parser.add_argument('-gt','--graphtype', metavar='Graph Type Directed/Undirected',type=str,choices=['directed','undirected'],default='undirected')
	args=parser.parse_args()
	if args.plottype=='roc':
		plot_roc(args.graphpath,args.fcgclass,args.fcgtype,args.graphtype,args.prefix)
	elif args.plottype=='dist':
		plot_dist(args.graphpath,args.fcgclass,args.fcgtype,args.graphtype,args.prefix)