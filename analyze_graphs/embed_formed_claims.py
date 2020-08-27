from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import re
import os
import csv
import networkx as nx
import argparse
import codecs
import json
from flatten_dict import flatten
from scipy.spatial.distance import cosine

def embed_formed_claims(rdf_path,graph_path,model_path,embed_path):
	os.makedirs(embed_path, exist_ok=True)
	model = SentenceTransformer(model_path)
	for source_fcg_type in ['tfcg','ffcg']:
		for target_claim_type in ['true','false']:
			#loading claims
			target_claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(target_claim_type)))
			#loading target claim embeddings
			target_claims_embed=pd.read_csv(os.path.join(embed_path,target_claim_type+"_claims_embeddings_({}).tsv".format(model_path.split("/")[-1])),delimiter="\t",header=None).values
			for mode in ['d','w']:
				write_path=os.path.join(graph_path,"fred","paths",source_fcg_type+"_"+target_claim_type+"_({})".format(model_path.split("/")[-1]),"paths_{}".format(mode))
				with codecs.open(write_path+".json","r","utf-8") as f: 
					paths=json.loads(f.read())
				flatten_paths=flatten(paths)
				formed_claims_keys=[f for f in list(flatten_paths.keys()) if 'formed_claim' in f]
				formed_claims=[flatten_paths[f] for f in formed_claims_keys]
				formed_claims_embed=model.encode(formed_claims)
				formed_claims_claimIDs=[int(f[0]) for f in formed_claims_keys]
				claimIXs=[target_claims[target_claims['claimID']==claimID].index[0] for claimID in formed_claims_claimIDs]
				ps=[np.array([target_claims_embed[claimIX]]) for claimIX in claimIXs]
				dists=[round(np.arccos(1-cosine(ps[i],formed_claims_embed[i]))/np.pi,3) for i in range(len(formed_claims_keys))]
				for i in range(len(formed_claims_keys)):
					u,v,w,d=eval(formed_claims_keys[i][1])
					new_key=str((u,v,w,d,dist[i]))
					paths[formed_claims_keys[i][0]][new_key] = paths[formed_claims_keys[i][0]][formed_claims_keys[i][1]]
				with codecs.open(write_path+"2.json","w","utf-8") as f: 
					f.write(json.dumps(paths,indent=5,ensure_ascii=False))

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Embed Fred Path Formed Claims')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/rdf_files/")
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/graphs/")
	parser.add_argument('-mp','--modelpath', metavar='model path',type=str,help='Model directory to load the model',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/models/claims-relatedness-model/claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27")
	parser.add_argument('-ep','--embedpath', metavar='embed path',type=str,help='Model directory to save and load embeddings',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/embeddings")
	parser.add_argument('-gt','--graphtype', metavar='Graph Type Directed/Undirected',type=str,choices=['directed','undirected'])
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	args=parser.parse_args()
	embed_formed_claims(args.rdfpath,args.graphpath,args.modelpath,args.embedpath)