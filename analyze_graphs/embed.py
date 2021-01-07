from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import os
import csv
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import argparse

def embed_claims(rdf_path,model_path,embed_path,claim_type):
	os.makedirs(embed_path, exist_ok=True)
	model = SentenceTransformer(model_path)
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims_og.csv".format(claim_type)),index_col=0)
	claims_list=list(claims['claim_text'])
	claims_embeddings=model.encode(claims_list)
	if model_path=="roberta-base-nli-stsb-mean-tokens":
		#Calculating angular distance
		claim_claim=1-np.arccos(np.clip(cosine_similarity(claims_embeddings,claims_embeddings),-1,1))/np.pi
		#storing index of duplicate-ish claims to get rid of them from the data
		np.fill_diagonal(claim_claim,np.nan)
		indices=np.argwhere(claim_claim>0.825)
		x_claim=list(set([np.sort(ind)[-1] for ind in indices]))
		claims=claims.drop(index=x_claim).reset_index()
		claims.to_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)),index=False)
		np.save(os.path.join(rdf_path,"{}_claimID.npy".format(claim_type)),list(claims["claimID"]))
		claims_embeddings=np.delete(claims_embeddings,x_claim,0)
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

def clean_node_labels(node_label):
	node_label=node_label.split(':')[-1]
	#regex to find words in strings like "IllegalAllien" and "Want_3210203" and "President_of_the_United_States"
	regex_words=re.compile(r'([A-Z][a-z]*|[a-z]+)')
	node_label=" ".join(regex_words.findall(node_label))
	return node_label

def embed_nodes(graph_path,model_path,embed_path,fcg_type,fcg_class,graph_type):
	os.makedirs(embed_path, exist_ok=True)
	model = SentenceTransformer(model_path)
	fcg=nx.read_edgelist(os.path.join(graph_path,fcg_class,fcg_type,"{}.edgelist".format(fcg_type)),comments="@",create_using=eval(graph_type))
	nodes_list=list(fcg.nodes)
	nodes_clean_list=list(map(clean_node_labels,nodes_list))
	nodes_embeddings=model.encode(nodes_clean_list)
	with open(os.path.join(embed_path,fcg_type+'_nodes_embeddings_({}).tsv'.format(model_path.split("/")[-1])),'w',newline='') as f:
		for vector in nodes_embeddings:
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)
	with open(os.path.join(embed_path,fcg_type+'_nodes_embeddings_labels_({}).tsv'.format(model_path.split("/")[-1])),'w',newline='') as f:
		vector=['node_label','node_clean_label']
		tsv_output=csv.writer(f,delimiter='\t')
		tsv_output.writerow(vector)
		for i in range(len(nodes_list)):
			vector=list((nodes_list[i],nodes_clean_list[i]))
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(vector)

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Embed Fred Nodes or Claims')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/rdf_files/")
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/graphs/")
	parser.add_argument('-mp','--modelpath', metavar='model path',type=str,help='Model directory to load the model',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/models/claims-relatedness-model/claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27")
	parser.add_argument('-ep','--embedpath', metavar='embed path',type=str,help='Model directory to save and load embeddings',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/embeddings")
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg','ffcg','tfcg_co','ffcg_co','ufcg','covid19'],help='True/False/Union/Covid19 FactCheckGraph')
	parser.add_argument('-fc','--fcgclass', metavar='FactCheckGraph class',type=str,choices=['co_occur','fred'],default='co_occur')
	parser.add_argument('-gt','--graphtype', metavar='Graph Type Directed/Undirected',type=str,choices=['directed','undirected'],default='undirected')
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	args=parser.parse_args()
	claim_types={'tfcg_co':'true','ffcg_co':'false','tfcg':'true','ffcg':'false'}
	graph_types={'undirected':'nx.MultiGraph','directed':'nx.MultiDiGraph'}
	if args.fcgclass=='fred':
		embed_nodes(args.graphpath,args.modelpath,args.embedpath,args.fcgtype,args.fcgclass,graph_types[args.graphtype])
	elif args.fcgclass=='co_occur':
		embed_claims(args.rdfpath,args.modelpath,args.embedpath,claim_types[args.fcgtype])