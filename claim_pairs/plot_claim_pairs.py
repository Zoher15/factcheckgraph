import pandas as pd
import argparse
import os 
import json
from sklearn import metrics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import codecs
import seaborn as sns

def compile_claims(pairs_path,kg_label,n):
	claim_types=['true','false']
	jobs_path=os.path.join(pairs_path,"jobs")
	claim_pairs_scores=pd.DataFrame()
	#Iterating over the split job outputs and creating a sigle dataframe
	for i in range(n):
		df=pd.read_json(os.path.join(jobs_path,"{}_claim_pairs_{}_job.json".format(str(i+1),kg_label)))
		claim_pairs_scores=claim_pairs_scores.append(df, ignore_index = True)
	#Loading the claim entity pairs in str format
	claim_pairs_list=np.loadtxt(os.path.join(pairs_path,"claim_pairs.txt"),delimiter=";",dtype=str,encoding='utf-8')
	#Loading the claim entity pair to ID dict
	with codecs.open(os.path.join(pairs_path,"true_claim_pairs2ID.json"),"r","utf-8") as f:
		true_claim_pairs2ID=json.loads(f.read())
	with codecs.open(os.path.join(pairs_path,"false_claim_pairs2ID.json"),"r","utf-8") as f:
		false_claim_pairs2ID=json.loads(f.read())
	#Finding the index and ID for each pair
	true_claim_pairs_indID=np.asmatrix([[i,true_claim_pairs2ID[pair]] for i,pair in enumerate(claim_pairs_list) if pair in set(list(true_claim_pairs2ID.keys()))])
	false_claim_pairs_indID=np.asmatrix([[i,false_claim_pairs2ID[pair]] for i,pair in enumerate(claim_pairs_list) if pair in set(list(false_claim_pairs2ID.keys()))])
	#Mapping the indices and ID respectively
	true_claim_pairs_ind=list(map(int,np.asarray(true_claim_pairs_indID[:,0].flatten())[0]))
	true_claim_pairs_ID=list(np.asarray(true_claim_pairs_indID[:,1].flatten())[0])
	false_claim_pairs_ind=list(map(int,np.asarray(false_claim_pairs_indID[:,0].flatten())[0]))
	false_claim_pairs_ID=list(np.asarray(false_claim_pairs_indID[:,1].flatten())[0])
	#Setting labels and IDs in the dataframe using the indices
	claim_pairs_scores.loc[true_claim_pairs_ind,"label"]=1
	claim_pairs_scores.loc[false_claim_pairs_ind,"label"]=0
	claim_pairs_scores.loc[true_claim_pairs_ind,"claimID"]=true_claim_pairs_ID
	claim_pairs_scores.loc[false_claim_pairs_ind,"claimID"]=false_claim_pairs_ID
	import pdb
	pdb.set_trace()
	for mode in ['min','max','all','mean']:
		claimscores=claim_pairs_scores.drop(columns=['sfid','sid','stitle','tfid','tid','ttitle','rating','rem','paths'])
		#Plotting Distribution Plot
		plt.figure()
		plot_path=os.path.join(pairs_path,"plots")
		title="distribution true vs false claim all pairs {} {}".format(kg_label,mode)
		if not mode=='all':
			claimscores=eval("claimscores.groupby(['claimID']).{}()".format(mode))
		plt.hist(claimscores[claimscores['label']==1]['simil'],density=True,histtype='step',label='true')
		plt.hist(claimscores[claimscores['label']==0]['simil'],density=True,histtype='step',label='true')
		# plt.title(title)
		# sns.set_style("white")
		# sns.set_style("ticks")
		# ax = sns.kdeplot(pd.Series(claimscores[claimscores['label']==1]['simil'],name="true claims"))
		# ax = sns.kdeplot(pd.Series(claimscores[claimscores['label']==0]['simil'],name="false claims"))
		plt.xlabel("Proximity Scores")
		plt.title("True, False Claim Pairs Proximity Distribution")
		plt.ylabel("Density")
		plt.legend(loc="upper right")
		plt.tight_layout()
		plt.savefig(os.path.join(plot_path,title.replace(" ","_")+".png"))
		plt.close()
		plt.clf()
	# plt.show()
	#Plotting for all pairs
	# title="true vs false claim all pairs {}".format(kg_label)
	# lw=2
	# y=list(claim_pairs_scores['label'])
	# scores=list(claim_pairs_scores['simil'])
	# fpr,tpr,thresholds=metrics.roc_curve(y,scores, pos_label=1)
	# plt.plot(fpr,tpr,lw=lw,label=' (%0.2f) '%metrics.auc(fpr,tpr))
	# plt.xlabel('true positive rate')
	# plt.ylabel('false positive rate')
	# plt.legend(loc="lower right")
	# plt.title(title)
	# plt.tight_layout()
	# os.makedirs(plot_path,exist_ok=True)
	# plt.savefig(os.path.join(plot_path,title.replace(" ","_")+".png"))
	# plt.close()
	# plt.clf()

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Plot claim pairs')
	parser.add_argument('-pp','--pairspath', metavar='pairs path',type=str,help='Directory for the claim pairs',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/claim_pairs/')
	parser.add_argument('-kg','--kgtype', metavar='knowledgegraph type',type=str,choices=['dbpedia','wikidata'],help='Choose KnowledgeGraph Type')
	parser.add_argument('-n','--splits', metavar='split number',type=int,help='Number of Splits')
	args=parser.parse_args()
	compile_claims(args.pairspath,args.kgtype,args.splits)