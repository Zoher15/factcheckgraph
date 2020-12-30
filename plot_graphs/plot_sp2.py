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
from statistics import mean

def aggregate_edge_data(evalues,mode):
	#mode can be dist or weight
	edgepair_weights=[]
	for edgepair,e2values in evalues.items():
		edgepair_weights.append(e2values[mode])
	return sum(edgepair_weights)

def aggregate_weights(claim_D,mode,mode2):
	#mode can be w d or f
	#mode2 can be max, min, sum, mean
	edge_weights=[]
	for edge,evalues in claim_D.items():
		if type(evalues)!=list:
			#edge looks like this: "('db:John_McCain', 'db:United_States_Senate', 0.09, 1.25)"
			#u is the source node,v is the target node, w the special similarity, d the 1/1+distance
			u,v,w,d=eval(edge.replace("inf","np.inf"))
			# u,v,w,d,f=eval(edge.replace("inf","np.inf"))
			edge_weights.append(eval(mode))
	return eval("{}(edge_weights)".format(mode2))

def domb(numlist):
	domb=numlist[0]
	for j in numlist[1:]:
		if domb==0 and j==0:
			domb=0
		else:
			domb=(domb*j)/(domb+j-(domb*j))
	return domb

def aggregate_weights_diff(graph_path,fcg_class,fcg_type,graph_type):
	fcg_types={"co_occur":{"tfcg":"tfcg_co","ffcg":"ffcg_co"},"fred":{"tfcg":"tfcg","ffcg":"ffcg"}}
	embed={'roberta-base-nli-stsb-mean-tokens':'e1'}#,'claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27':'e2'}
	mode={'w':'d1','d':'d2'}#,'f':'d3'}
	aggmode={'mean':'a1','max':'a2'}#,'min':'a3'}#,'domb':'a4'}
	read_path_tfcg=os.path.join(graph_path,fcg_class,"paths",fcg_types[fcg_class]['tfcg'])
	read_path_ffcg=os.path.join(graph_path,fcg_class,"paths",fcg_types[fcg_class]['ffcg'])
	for e in list(embed.keys()):
		for d in list(mode.keys()):
			for a in list(aggmode.keys()):
				with codecs.open(os.path.join(read_path_tfcg+"_true_({})".format(e),"paths_"+graph_type+"_"+d+"_"+a+".json"),"r","utf-8") as f: 
					true_paths_tfcg=json.loads(f.read())
				with codecs.open(os.path.join(read_path_tfcg+"_false_({})".format(e),"paths_"+graph_type+"_"+d+"_"+a+".json"),"r","utf-8") as f: 
					false_paths_tfcg=json.loads(f.read())
				with codecs.open(os.path.join(read_path_ffcg+"_true_({})".format(e),"paths_"+graph_type+"_"+d+"_"+a+".json"),"r","utf-8") as f: 
					true_paths_ffcg=json.loads(f.read())
				with codecs.open(os.path.join(read_path_ffcg+"_false_({})".format(e),"paths_"+graph_type+"_"+d+"_"+a+".json"),"r","utf-8") as f: 
					false_paths_ffcg=json.loads(f.read())
				#Resetting tfcg and ffcg dictionaries so that we can compare them on the same keys: claimID
				true_scores_tfcg={eval(t[0])[0]:{(str(eval(s[0])[:2]) if s[0]!='target_claim' else s[0]):({'tfcg':{s[0]:s[1]}} if s[0]!='target_claim' else s[1]) for s in t[1].items()} for t in true_paths_tfcg.items()}
				false_scores_tfcg={eval(t[0])[0]:{(str(eval(s[0])[:2]) if s[0]!='target_claim' else s[0]):({'tfcg':{s[0]:s[1]}} if s[0]!='target_claim' else s[1]) for s in t[1].items()} for t in false_paths_tfcg.items()}
				true_scores_ffcg={eval(t[0])[0]:{(str(eval(s[0])[:2]) if s[0]!='target_claim' else s[0]):({'ffcg':{s[0]:s[1]}} if s[0]!='target_claim' else s[1]) for s in t[1].items()} for t in true_paths_ffcg.items()}
				false_scores_ffcg={eval(t[0])[0]:{(str(eval(s[0])[:2]) if s[0]!='target_claim' else s[0]):({'ffcg':{s[0]:s[1]}} if s[0]!='target_claim' else s[1]) for s in t[1].items()} for t in false_paths_ffcg.items()}
				#finding claimIDs that may not be have been checked by the other 
				true_miss_claimIDs=(set(true_scores_tfcg.keys())-set(true_scores_ffcg.keys())).union(set(true_scores_ffcg.keys())-set(true_scores_tfcg.keys()))
				false_miss_claimIDs=(set(false_scores_tfcg.keys())-set(false_scores_ffcg.keys())).union(set(false_scores_ffcg.keys())-set(false_scores_tfcg.keys()))
				import pdb
				pdb.set_trace()
				#setting the scores for missing claimIDs to 0
				#For claimIDs that are missing, emulate the edges present in the other (tfcg/ffcg) dictionary and set the items to (u,v,0,0):{}
				true_scores_tfcg_miss={t:{str(eval(s[0])[:2] if s[0]!='target_claim' else s[0]):({'tfcg':{str(eval(s[0])[:2]+(0,0)):{}}} if s[0]!='target_claim' else s[1]) for s in true_scores_ffcg[t].items()} for t in true_miss_claimIDs if t not in set(true_scores_tfcg.keys())}
				true_scores_ffcg_miss={t:{str(eval(s[0])[:2] if s[0]!='target_claim' else s[0]):({'ffcg':{str(eval(s[0])[:2]+(0,0)):{}}} if s[0]!='target_claim' else s[1]) for s in true_scores_tfcg[t].items()} for t in true_miss_claimIDs if t not in set(true_scores_ffcg.keys())}
				false_scores_tfcg_miss={t:{str(eval(s[0])[:2] if s[0]!='target_claim' else s[0]):({'tfcg':{str(eval(s[0])[:2]+(0,0)):{}}} if s[0]!='target_claim' else s[1]) for s in false_scores_ffcg[t].items()} for t in false_miss_claimIDs if t not in set(false_scores_tfcg.keys())}
				false_scores_ffcg_miss={t:{str(eval(s[0])[:2] if s[0]!='target_claim' else s[0]):({'ffcg':{str(eval(s[0])[:2]+(0,0)):{}}} if s[0]!='target_claim' else s[1]) for s in false_scores_tfcg[t].items()} for t in false_miss_claimIDs if t not in set(false_scores_ffcg.keys())}
				#updating original dictionaries
				true_scores_tfcg.update(true_scores_tfcg_miss)
				false_scores_tfcg.update(false_scores_tfcg_miss)
				true_scores_ffcg.update(true_scores_ffcg_miss)
				false_scores_ffcg.update(false_scores_ffcg_miss)
				#True Scores will be the final combined dict incorporating both the tfcg and ffcg paths with the respective values of each edge i.e. the difference between the tfcg and ffcg score
				true_scores={}
				try:
					for i in set(true_scores_tfcg.keys()).union(set(true_scores_ffcg.keys())):
						d_list=[]
						w_list=[]
						temp={}
						for k in list(true_scores_tfcg[i].keys()):
							if k!='target_claim':
								tu,tv,tw,td=eval(list(true_scores_tfcg[i][k]['tfcg'].keys())[0])
								fu,fv,fw,fd=eval(list(true_scores_ffcg[i][k]['ffcg'].keys())[0])
								w=tw-fw
								d=td-fd
								temp[str((tu,tv,w,d))]={'tfcg':true_scores_tfcg[i][k]['tfcg'],'ffcg':true_scores_ffcg[i][k]['ffcg']}
								d_list.append(d)
								w_list.append(w)
							else:
								temp[k]=true_scores_tfcg[i][k]
				#mode can be w d or f
				#aggmode can be max, min, sum, mean
				edge_weights=[]
				for edge,evalues in claim_D.items():
					if type(evalues)!=list:
						#edge looks like this: "('db:John_McCain', 'db:United_States_Senate', 0.09, 1.25)"
						#u is the source node,v is the target node, w the special similarity, d the 1/1+distance
						u,v,w,d=eval(edge.replace("inf","np.inf"))
						# u,v,w,d,f=eval(edge.replace("inf","np.inf"))
						edge_weights.append(eval(mode))
				return eval("{}(edge_weights)".format(aggmode))


def plot_roc(graph_path,fcg_class,fcg_type,graph_type):
	fcg_types={"co_occur":{"tfcg":"tfcg_co","ffcg":"ffcg_co"},"fred":{"tfcg":"tfcg","ffcg":"ffcg"}}
	embed={'roberta-base-nli-stsb-mean-tokens':'e1'}#,'claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27':'e2'}
	mode={'w':'d1','d':'d2'}#,'f':'d3'}
	aggmode={'mean':'a1','max':'a2'}#,'min':'a3'}#,'domb':'a4'}
	read_path_tfcg=os.path.join(graph_path,fcg_class,"paths",fcg_types[fcg_class]['tfcg'])
	read_path_ffcg=os.path.join(graph_path,fcg_class,"paths",fcg_types[fcg_class]['ffcg'])
	plot_path=os.path.join(graph_path,fcg_class,"plots")
	plot_dict={}
	plot_dict['roc']={}
	plot_dict['pr']={}
	plot_dict['f1']={}
	for e in list(embed.keys()):
		for d in list(mode.keys()):
			for a in list(aggmode.keys()):
				with codecs.open(os.path.join(read_path_tfcg+"_true_({})".format(e),"paths_"+graph_type+"_"+d+"_"+a+".json"),"r","utf-8") as f: 
					true_paths_tfcg=json.loads(f.read())
				with codecs.open(os.path.join(read_path_tfcg+"_false_({})".format(e),"paths_"+graph_type+"_"+d+"_"+a+".json"),"r","utf-8") as f: 
					false_paths_tfcg=json.loads(f.read())
				with codecs.open(os.path.join(read_path_ffcg+"_true_({})".format(e),"paths_"+graph_type+"_"+d+"_"+a+".json"),"r","utf-8") as f: 
					true_paths_ffcg=json.loads(f.read())
				with codecs.open(os.path.join(read_path_ffcg+"_false_({})".format(e),"paths_"+graph_type+"_"+d+"_"+a+".json"),"r","utf-8") as f: 
					false_paths_ffcg=json.loads(f.read())
				#creating a dictionary with the claimID as key and score as value
				aggregate_weights_diff(true_paths_tfcg,true_paths_ffcg,false_paths_tfcg,false_paths_ffcg,d,a)

				#tfcg score- ffcg score for each claim
				true_scores=[true_scores_tfcg[i]-true_scores_ffcg[i] for i in set(true_scores_tfcg.keys()).union(set(true_scores_ffcg.keys()))]
				false_scores=[false_scores_tfcg[i]-false_scores_ffcg[i] for i in set(false_scores_tfcg.keys()).union(set(false_scores_ffcg.keys()))] 
				true_y=[1 for i in range(len(true_scores))]
				false_y=[0 for i in range(len(false_scores))]
				y=true_y+false_y
				scores=true_scores+false_scores
				fpr,tpr,thresholds=metrics.roc_curve(y,scores, pos_label=1)
				precision,recall,thresholds=metrics.precision_recall_curve(y,scores, pos_label=1)
				f1scores=[2*(precision[i]*recall[i])/(precision[i]+recall[i]) for i in range(len(thresholds))]
				mlabel=embed[e]+mode[d]+aggmode[a]
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
				plot_dict['f1'][mlabel]['label']=mlabel+' AVG_Pr (%0.2f) '%metrics.average_precision_score(y,scores)
	#Plotting
	lw=2
	title=graph_type+" {} shortest path embedded paths".format(fcg_class+"_"+fcg_type)
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

def plot_dist(graph_path,fcg_class,fcg_type,graph_type):
	fcg_types={"co_occur":{"tfcg":"tfcg_co","ffcg":"ffcg_co"},"fred":{"tfcg":"tfcg","ffcg":"ffcg"}}
	embed={'roberta-base-nli-stsb-mean-tokens':'e1'}#,'claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27':'e2'}
	mode={'w':'d1','d':'d2'}#,'f':'d3'}
	aggmode={'mean':'a1','max':'a2'}#,'min':'a3'}#,'domb':'a4'}
	read_path_tfcg=os.path.join(graph_path,fcg_class,"paths",fcg_types[fcg_class]['tfcg'])
	read_path_ffcg=os.path.join(graph_path,fcg_class,"paths",fcg_types[fcg_class]['ffcg'])
	plot_path=os.path.join(graph_path,fcg_class,"plots")
	for e in list(embed.keys()):
		for d in list(mode.keys()):
			for a in list(aggmode.keys()):
				label=embed[e]+mode[d]+aggmode[a]
				plt.figure()
				title=graph_type+" density histogram {} shortest path setence embedded paths".format(fcg_class+"_"+fcg_type+"_"+label)
				with codecs.open(os.path.join(read_path_tfcg+"_true_({})".format(e),"paths_"+graph_type+"_"+d+"_"+a+".json"),"r","utf-8") as f: 
					true_paths_tfcg=json.loads(f.read())
				with codecs.open(os.path.join(read_path_tfcg+"_false_({})".format(e),"paths_"+graph_type+"_"+d+"_"+a+".json"),"r","utf-8") as f: 
					false_paths_tfcg=json.loads(f.read())
				with codecs.open(os.path.join(read_path_ffcg+"_true_({})".format(e),"paths_"+graph_type+"_"+d+"_"+a+".json"),"r","utf-8") as f: 
					true_paths_ffcg=json.loads(f.read())
				with codecs.open(os.path.join(read_path_ffcg+"_false_({})".format(e),"paths_"+graph_type+"_"+d+"_"+a+".json"),"r","utf-8") as f: 
					false_paths_ffcg=json.loads(f.read())
				#creating a dictionary with the claimID as key and score as value
				true_scores_tfcg={eval(t[0])[0]:eval(t[0])[1] for t in true_paths_tfcg.items()}
				false_scores_tfcg={eval(t[0])[0]:eval(t[0])[1] for t in false_paths_tfcg.items()}
				true_scores_ffcg={eval(t[0])[0]:eval(t[0])[1] for t in true_paths_ffcg.items()}
				false_scores_ffcg={eval(t[0])[0]:eval(t[0])[1] for t in false_paths_ffcg.items()}
				#finding claimIDs that may not be have been checked by the other 
				true_miss_claimIDs=(set(true_scores_tfcg.keys())-set(true_scores_ffcg.keys())).union(set(true_scores_ffcg.keys())-set(true_scores_tfcg.keys()))
				false_miss_claimIDs=(set(false_scores_tfcg.keys())-set(false_scores_ffcg.keys())).union(set(false_scores_ffcg.keys())-set(false_scores_tfcg.keys()))
				#setting the scores for missing claimIDs to 0
				true_scores_tfcg_miss={t:0 for t in true_miss_claimIDs if t not in set(true_scores_tfcg.keys())}
				true_scores_ffcg_miss={t:0 for t in true_miss_claimIDs if t not in set(true_scores_ffcg.keys())}
				false_scores_tfcg_miss={t:0 for t in false_miss_claimIDs if t not in set(false_scores_tfcg.keys())}
				false_scores_ffcg_miss={t:0 for t in false_miss_claimIDs if t not in set(false_scores_ffcg.keys())}
				#updating original dictionaries
				true_scores_tfcg.update(true_scores_tfcg_miss)
				false_scores_tfcg.update(false_scores_tfcg_miss)
				true_scores_ffcg.update(true_scores_ffcg_miss)
				false_scores_ffcg.update(false_scores_ffcg_miss)
				#tfcg score- ffcg score for each claim
				true_scores=[true_scores_tfcg[i]-true_scores_ffcg[i] for i in set(true_scores_tfcg.keys()).union(set(true_scores_ffcg.keys()))]
				false_scores=[false_scores_tfcg[i]-false_scores_ffcg[i] for i in set(false_scores_tfcg.keys()).union(set(false_scores_ffcg.keys()))] 
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
	parser.add_argument('-ft','--fcgtype', metavar='FCG Type',type=str,choices=['tfcg','ffcg','tfcg_co','ffcg_co'])
	parser.add_argument('-pt','--plottype', metavar='plot type',type=str,choices=['roc','dist'],help='Class of graph to plot')
	parser.add_argument('-gt','--graphtype', metavar='Graph Type Directed/Undirected',type=str,choices=['directed','undirected'])
	args=parser.parse_args()
	if args.plottype=='roc':
		plot_roc(args.graphpath,args.fcgclass,args.fcgtype,args.graphtype)
	elif args.plottype=='dist':
		plot_dist(args.graphpath,args.fcgclass,args.fcgtype,args.graphtype)