from scipy.stats import pearsonr,kendalltau,spearmanr
import pandas as pd
import numpy as np
import argparse

def calc_corr(graphpath,fcg_class,fcg_label,kg_label,start,end,corr_type):
	if fcg_class=="co-occur":
		fcg_label=fcg_label+"_co"
	elif fcg_class=="backbone":
		fcg_label=fcg_label+"_bb"
	kg_scores=list(np.load(os.path.join(graphpath,"kg",kg_label,"{}_scores.npy".format(kg_label))))
	fcg_scores=list(np.load(os.path.join(graphpath,fcg_class,fcg_label,"{}_scores.npy".format(fcg_label))))
	scores=pd.DataFrame(columns=[kg_label,fcg_label])
	scores[kg_label]=kg_scores
	scores[fcg_label]=fcg_scores
	scores=scores.sort_values(by=kg_label,ascending=False).reset_index(drop=True)
	if end>len(scores) or end==None:
		end=len(scores_all)
	correlations=pd.DataFrame(columns=['index','type',corr_type,'p-value'])
	for i in range(start,end):
		kg_fcg=eval(corr_type)(scores.loc[0:i+1,kg_label],scores_all.loc[0:i+1,fcg_label])
		kg_fcg_row={'index':i+1,'type':'{} - {}'.format(kg_label,fcg_label),corr_type:kg_fcg[0],'p-value':kg_fcg[1]}
		correlations=correlations.append(kg_fcg_row,ignore_index=True)
	corr_path=os.path.join(graphpath,fcg_class,fcg_label,"corr")
	os.makedirs(corr_path,exist_ok=True)
	if start==0 and end==len(scores):
		correlations.to_csv(os.path.join(corr_path,"corr_{}_{}.csv".format(kg_label,fcg_label)),index=False)
	else:
		correlations.to_csv(os.path.join(corr_path,"{}_{}_corr_{}_{}.csv".format(start,end,kg_label,fcg_label)),index=False)

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Find proximity correlations on pairs of entities in knowledge graphs')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	parser.add_argument('-fcg','--fcgclass', metavar='fcg class',type=str,choices=['fred','co-occur','backbone'],help='Class of FactCheckGraph to process')
	parser.add_argument('-fcg','--fcgtype', metavar='fcg type',type=str,choices=['tfcg','ffcg','ufcg'],help='Type of FactCheckGraph to process')
	parser.add_argument('-kg','--kgtype', metavar='knowledgegraph type',type=str,choices=['dbpedia','wikidata'],help='DBPedia or Wikidata Graph')
	parser.add_argument('-s','--start', metavar='start index',type=int,help='Index of all pairs to start from',default=0)
	parser.add_argument('-e','--end', metavar='end index',type=int,help='Index of all pairs to end at',default=None)
	parser.add_argument('-c','--corr', metavar='correlation type',type=str,choices=['kendalltau','spearmanr','pearsonr'],help='Type of correlation')
	args=parser.parse_args()
	calc_corr(args.graphpath,args.fcgclass,args.kgtype,args.start,args.end,args.corr)