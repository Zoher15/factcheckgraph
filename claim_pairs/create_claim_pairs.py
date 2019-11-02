import os
import pandas as pd
import networkx as nx
import re
import numpy as np
import argparse
import rdflib
import codecs
import json
from itertools import combinations

def create_claim_scores(rdf_path,graph_path,pairs_path,kg_label):
	claim_types=['true','false']
	#Set for storing pairs
	true_claim_pairs_set=set([])
	false_claim_pairs_set=set([])
	#Dict for storing claimID:list(pairs)
	true_claim_pairs={}
	false_claim_pairs={}
	true_claim_entities={}
	false_claim_entities={}
	#Dict for storing str(pair):claimID
	true_claim_pairs2ID={}
	false_claim_pairs2ID={}
	kg_path=os.path.join(graph_path,"kg",kg_label,"data",kg_label)
	#Fetching kg entities
	kg_entities=set(list(np.loadtxt(kg_path+"_entities.txt",dtype='str',encoding='utf-8')))
	for claim_type in claim_types:
		claim_IDs=np.load(os.path.join(rdf_path,"{}_claimID.npy".format(claim_type)))
		entity_regex=re.compile(r'http:\/\/dbpedia\.org')
		for claim_ID in claim_IDs:
			claim_entities_set=set([])
			claim_g=rdflib.Graph()
			filename="claim{}.rdf".format(str(claim_ID))
			try:
				claim_g.parse(os.path.join(rdf_path,"{}_claims".format(claim_type),filename),format='application/rdf+xml')
			except:
				continue
			for triple in claim_g:
				subject,predicate,obj=list(map(str,triple))
				try:
					if entity_regex.search(subject) and subject in kg_entities:
						claim_entities_set.add(subject)
					if entity_regex.search(obj) and obj in kg_entities:
						claim_entities_set.add(obj)
				except KeyError:
					pass
			eval(claim_type+"_claim_entities")[str(claim_ID)]=list(claim_entities_set)
			eval(claim_type+"_claim_pairs")[str(claim_ID)]=set(map(lambda x:str(tuple(sorted(x))),list(combinations(eval(claim_type+"_claim_entities")[str(claim_ID)],2))))
			for pair in eval(claim_type+"_claim_pairs")[str(claim_ID)]:
				eval(claim_type+"_claim_pairs_set").add(pair)
				eval(claim_type+"_claim_pairs2ID")[pair]=str(claim_ID)
	intersect_claim_pairs_set=true_claim_pairs_set.intersection(false_claim_pairs_set)
	#Deleting pairs that are common to both sets:
	for pair in intersect_claim_pairs_set:
		true_claim_pairs[true_claim_pairs2ID[pair]].remove(pair)
		false_claim_pairs[false_claim_pairs2ID[pair]].remove(pair)
		if len(true_claim_pairs[true_claim_pairs2ID[pair]])==0:
			del true_claim_pairs[true_claim_pairs2ID[pair]]
		if len(false_claim_pairs[false_claim_pairs2ID[pair]])==0:
			del false_claim_pairs[false_claim_pairs2ID[pair]]
		del true_claim_pairs2ID[pair]
		del false_claim_pairs2ID[pair]
	#Saving the dicts
	for claim_type in claim_types:
		claim_pairs={key:list(value) for key,value in eval("{}_claim_pairs".format(claim_type)).items()}
		with codecs.open(os.path.join(pairs_path,"{}_claim_pairs.json".format(claim_type)),"w","utf-8") as f:
			f.write(json.dumps(claim_pairs,ensure_ascii=False))
		with codecs.open(os.path.join(pairs_path,"{}_claim_pairs2ID.json".format(claim_type)),"w","utf-8") as f:
			f.write(json.dumps(eval("{}_claim_pairs2ID".format(claim_type)),ensure_ascii=False))
	#Creating list of pairs
	true_claim_pairs_list=list(true_claim_pairs2ID.keys())
	false_claim_pairs_list=list(false_claim_pairs2ID.keys())
	claim_pairs_list=true_claim_pairs_list+false_claim_pairs_list
	#Saving the pairs in the str(tuple) format in the same index as the list
	with codecs.open(os.path.join(pairs_path,"claim_pairs.txt"),"w","utf-8") as f:
		for pair in claim_pairs_list:
			f.write("{}\n".format(pair))
	claim_pairs_list=list(map(eval,claim_pairs_list))
	#Reading the kg node2ID dict
	with codecs.open(kg_path+"_node2ID.json","r","utf-8") as f:
		node2ID=json.loads(f.read())
	#Creating the klformat
	claim_pairs_list_klformat=np.asarray([[np.nan,i[0],np.nan,np.nan,i[1],np.nan,np.nan] for i in claim_pairs_list])
	with codecs.open(os.path.join(pairs_path,"claim_pairs_{}_klformat.txt".format(kg_label)),"w","utf-8") as f:
		for line in claim_pairs_list_klformat:
			f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(node2ID[line[1]])),str(line[2]),str(line[3]),str(int(node2ID[line[4]])),str(line[5]),str(line[6])))	

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create co-cccur graph')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/rdf_files/')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	parser.add_argument('-pp','--pairspath', metavar='pairs path',type=str,help='True False or Union FactCheckGraph',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/claim_pairs/')
	parser.add_argument('-kg','--kg', metavar='knowledgegraph type',type=str,choices=['dbpedia','wikidata'],help='DBPedia or Wikidata Graph')
	args=parser.parse_args()
	create_claim_scores(args.rdfpath,args.graphpath,args.pairspath,args.kg)