import argparse
import pandas as pd
import numpy as np 
from sklearn import metrics
import json
from collections import OrderedDict
import os
import datetime
import codecs
from statistics import mean,median

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

def create_ordered_paths(rdf_path,graph_path,graph_type,fcg_class,fcg_type):
	fcg_types={"co_occur":{"tfcg":"tfcg_co","ffcg":"ffcg_co"},"fred":{"tfcg":"tfcg","ffcg":"ffcg","ufcg":"ufcg"}}
	embeds={'roberta-base-nli-stsb-mean-tokens':'e1'}#,'claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27':'e2'}
	dists={'w':'d1','d':'d2'}#,'f':'d3'}
	aggs={'mean':'a1','max':'a2'}#,'median':'a3'}#,'domb':'a4'}
	fcg_type=fcg_types[fcg_class][fcg_type]
	for embed in list(embeds.keys()):
		for dist in list(dists.keys()):
			for truth_label in ['true','false']:
				write_path=os.path.join(graph_path,fcg_class,"paths",truth_label+"_"+fcg_type+"_({})".format(embed.split("/")[-1]))
				write_path=os.path.join(write_path,"paths")
				write_path=write_path+"_"+graph_type
				#mode can be weight/dist
				rw_path=write_path+"_"+dist
				with codecs.open(rw_path+".json","r","utf-8") as f: 
					paths=json.loads(f.read())
				np.save(rw_path+"_claimIDs.npy",list(paths.keys()))
				for agg in list(aggs.keys()):
					ordered_paths={str((int(t[0]),aggregate_weights(t[1],dist,agg))):t[1] for t in paths.items()}
					ordered_paths=sorted(ordered_paths.items(), key=lambda t: eval(t[0])[1],reverse=True)
					top10p=ordered_paths[:int(len(ordered_paths)*.10)]
					bot10p=ordered_paths[int(len(ordered_paths)*.90):]
					ordered_paths=OrderedDict(ordered_paths)
					with codecs.open(rw_path+"_"+agg+".json","w","utf-8") as f:
						f.write(json.dumps(ordered_paths,indent=5,ensure_ascii=False))
					with codecs.open(rw_path+"_"+agg+"_1.json","w","utf-8") as f:
						f.write(json.dumps(OrderedDict(top10p),indent=5,ensure_ascii=False))
					with codecs.open(rw_path+"_"+agg+"_0.json","w","utf-8") as f:
						f.write(json.dumps(OrderedDict(bot10p),indent=5,ensure_ascii=False))

def create_ordered_paths_diff(rdf_path,graph_path,graph_type,fcg_class):
	fcg_types={"co_occur":{"tfcg":"tfcg_co","ffcg":"ffcg_co"},"fred":{"tfcg":"tfcg","ffcg":"ffcg"}}
	embeds={'roberta-base-nli-stsb-mean-tokens':'e1'}#,'claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27':'e2'}
	dists={'w':'d1','d':'d2'}#,'f':'d3'}
	aggs={'mean':'a1','max':'a2'}#,'median':'a3'}#,'domb':'a4'}
	for embed in list(embeds.keys()):
		for dist in list(dists.keys()):
			for truth_label in ['true','false']:
				#tfcg
				tfcg_type=fcg_types[fcg_class]['tfcg']
				write_path=os.path.join(graph_path,fcg_class,"paths",truth_label+"_"+tfcg_type+"_({})".format(embed.split("/")[-1]))
				write_path=os.path.join(write_path,"paths")
				write_path=write_path+"_"+graph_type
				#mode can be weight/dist
				rw_path=write_path+"_"+dist
				with codecs.open(rw_path+".json","r","utf-8") as f: 
					tfcg_paths=json.loads(f.read())
				np.save(rw_path+"_claimIDs.npy",list(tfcg_paths.keys()))
				#ffcg
				ffcg_type=fcg_types[fcg_class]['ffcg']
				write_path=os.path.join(graph_path,fcg_class,"paths",truth_label+"_"+ffcg_type+"_({})".format(embed.split("/")[-1]))
				write_path=os.path.join(write_path,"paths")
				write_path=write_path+"_"+graph_type
				#mode can be weight/dist
				rw_path=write_path+"_"+dist
				with codecs.open(rw_path+".json","r","utf-8") as f: 
					ffcg_paths=json.loads(f.read())
				np.save(rw_path+"_claimIDs.npy",list(ffcg_paths.keys()))
				#transforming tfcg and ffcg paths
				tfcg_paths={eval(claimID):{(str(eval(edge)[:2]) if edge!='target_claim' else edge):({'tfcg':{edge:edge_paths}} if edge!='target_claim' else edge_paths) for edge,edge_paths in claim_edges.items()} for claimID,claim_edges in tfcg_paths.items()}
				ffcg_paths={eval(claimID):{(str(eval(edge)[:2]) if edge!='target_claim' else edge):({'ffcg':{edge:edge_paths}} if edge!='target_claim' else edge_paths) for edge,edge_paths in claim_edges.items()} for claimID,claim_edges in ffcg_paths.items()}
				#Diff Calculate
				scores={}
				if len(set(tfcg_paths.keys())-set(ffcg_paths.keys()))==0 and len(set(ffcg_paths.keys())-set(tfcg_paths.keys()))==0:
					claimIDs=set(tfcg_paths.keys())
					for claimID in claimIDs:
						d_list=[]
						w_list=[]
						temp={}
						edges=set(tfcg_paths[claimID].keys())
						for edge in edges:
							if edge!='target_claim':
								tu,tv,tw,td=eval(list(tfcg_paths[claimID][edge]['tfcg'].keys())[0])
								fu,fv,fw,fd=eval(list(ffcg_paths[claimID][edge]['ffcg'].keys())[0])
								w=round(tw-fw,3)
								d=round(td-fd,3)
								temp[str((tu,tv,w,d))]={'tfcg':tfcg_paths[claimID][edge]['tfcg'],'ffcg':ffcg_paths[claimID][edge]['ffcg']}
								d_list.append(d)
								w_list.append(w)
							else:
								temp[str(edge)]=tfcg_paths[claimID][edge]
						scores[str(claimID)]=temp
				scores_path=os.path.join(graph_path,fcg_class,"paths",truth_label+"_scores_"+dist+"_({})".format(embed))
				with codecs.open(scores_path+".json","w","utf-8") as f:
					f.write(json.dumps(scores,indent=6,ensure_ascii=False))
				np.save(scores_path+"_claimIDs.npy",list(scores.keys()))
				for agg in list(aggs.keys()):
					ordered_paths={str((int(t[0]),aggregate_weights(t[1],dist,agg))):t[1] for t in scores.items()}
					ordered_paths=sorted(ordered_paths.items(), key=lambda t: eval(t[0])[1],reverse=True)
					top10p=ordered_paths[:int(len(ordered_paths)*.10)]
					bot10p=ordered_paths[int(len(ordered_paths)*.90):]
					ordered_paths=OrderedDict(ordered_paths)
					with codecs.open(scores_path+"_"+agg+".json","w","utf-8") as f:
						f.write(json.dumps(ordered_paths,indent=5,ensure_ascii=False))
					with codecs.open(scores_path+"_"+agg+"_1.json","w","utf-8") as f:
						f.write(json.dumps(OrderedDict(top10p),indent=5,ensure_ascii=False))
					with codecs.open(scores_path+"_"+agg+"_0.json","w","utf-8") as f:
						f.write(json.dumps(OrderedDict(bot10p),indent=5,ensure_ascii=False))

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Plotting true(adjacent) pairs vs false (non-adjacent)')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/rdf_files/")
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-fcg','--fcgclass', metavar='fcg class',type=str,help='Class of FactCheckGraph to process')
	parser.add_argument('-ft','--fcgtype', metavar='FCG Type',type=str,choices=['tfcg','ffcg','tfcg_co','ffcg_co','ufcg'])
	parser.add_argument('-gt','--graphtype', metavar='Graph Type Directed/Undirected',type=str,choices=['directed','undirected'],default='undirected')
	args=parser.parse_args()
	if args.fcgtype==None:
		create_ordered_paths_diff(args.rdfpath,args.graphpath,args.graphtype,args.fcgclass)
	else:
		create_ordered_paths(args.rdfpath,args.graphpath,args.graphtype,args.fcgclass,args.fcgtype)