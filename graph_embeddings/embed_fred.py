import networkx as nx
import pandas as pd 
import numpy as np
import argparse
import rdflib
import re
import html
import os
import csv
import codecs
from sentence_transformers import SentenceTransformer

def embed_nodes(graph_path,fcg_label,compilefred):
	read_path=os.path.join(graph_path,"fred"+str(compilefred),fcg_label+str(compilefred),str(fcg_label)+"{}.edgelist".format(str(compilefred)))
	G=nx.read_edgelist(read_path,comments="@")
	regex_vn=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/vn\/data\/([a-zA-Z]*)_.*')
	regex_dbpedia=re.compile(r'^http:\/\/dbpedia\.org\/resource\/(.*)')
	regex_fred=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#([a-zA-Z]*)_.*')
	regex_fredup=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#(.*)')
	regex_percent=re.compile(r'.*%*.*')
	vn=[]
	dbpedia=[]
	fred=[]
	fredup=[]
	others=[]
	nodes=list(G.nodes)
	print(len(nodes))
	print(len(set(nodes)))
	for node in nodes:
		if regex_dbpedia.match(node):
			dbpedia.append(regex_dbpedia.match(html.unescape(node))[1].replace("_"," "))
		elif regex_fred.match(node):
			fred.append(regex_fred.match(html.unescape(node))[1].replace("_"," "))
		elif regex_fredup.match(node):
			fredup.append(regex_fredup.match(html.unescape(node))[1].replace("_"," "))
		elif regex_vn.match(node):
			vn.append(regex_vn.match(html.unescape(node))[1].replace("_"," "))
		else:
			others.append(node)
	nset=dbpedia+fred+fredup+vn
	print(len(nset))
	print(len(set(nset)))
	model = SentenceTransformer('bert-base-nli-mean-tokens')
	fred_embeddings=model.encode(fred)
	dbpedia_embeddings=model.encode(dbpedia)
	vn_embeddings=model.encode(vn)
	with open('dbpedia_embeddings.tsv','w',newline='') as f1:
		with open('dbpedia_labels.tsv','w',newline='') as f:
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(['text'])
			for i in range(len(dbpedia)):
				tsv_output=csv.writer(f,delimiter='\t')
				tsv_output.writerow([dbpedia[i]])
				tsv_output=csv.writer(f1,delimiter='\t')
				tsv_output.writerow(dbpedia_embeddings[i])

	with open('vn_embeddings.tsv','w',newline='') as f1:
		with open('vn_labels.tsv','w',newline='') as f:
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(['text'])
			for i in range(len(vn)):
				tsv_output=csv.writer(f,delimiter='\t')
				tsv_output.writerow([vn[i]])
				tsv_output=csv.writer(f1,delimiter='\t')
				tsv_output.writerow(vn_embeddings[i])

	with open('fred_embeddings.tsv','w',newline='') as f1:
		with open('fred_labels.tsv','w',newline='') as f:
			tsv_output=csv.writer(f,delimiter='\t')
			tsv_output.writerow(['text'])
			for i in range(len(fred)):
				tsv_output=csv.writer(f,delimiter='\t')
				tsv_output.writerow([fred[i]])
				tsv_output=csv.writer(f1,delimiter='\t')
				tsv_output.writerow(fred_embeddings[i])

	with codecs.open("fredup.txt","w","utf-8") as f:
		for node in fredup:
			f.write(node+"\n")


if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Embed fred graph')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to retrieve the graphs from',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg','ffcg','ufcg'],help='True False or Union FactCheckGraph')
	parser.add_argument('-cf','--compilefred',metavar='Compile method #',type=int,help='Number of compile method',default=0)
	args=parser.parse_args()
	embed_nodes(args.graphpath,args.fcgtype,args.compilefred)