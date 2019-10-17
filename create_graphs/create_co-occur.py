import os
import pandas as pd
import networkx as nx
import re
import argparse
import rdflib

def create_co_occur(rdf_path,graph_path,fcg_type):
	fcg_labels={"true":"TFCG_co","false":"FFCG_co","union":"UFCG_co"}
	fcg_label=fcg_labels[fcg_type]
	if fcg_type=="union":
		#Assumes that TFCG_co and FFCG_co exists
		tfcgpath=os.path.join(graph_path,"TFCG_co","TFCG_co.edgelist")
		ffcgpath=os.path.join(graph_path,"FFCG_co","FFCG_co.edgelist")
		if os.path.exists(tfcgpath) and os.path.exists(ffcgpath):
			tfcg=nx.read_edgelist(tfcgpath)
			ffcg=nx.read_edgelist(ffcgpath)
			ufcg=nx.compose(tfcg,ffcg)
			writepath=os.path.join(graph_path,"UFCG_co")
			os.makedirs(writepath, exist_ok=True)
			nx.write_edgelist(FCG,os.path.join(writepath,"UFCG_co.edgelist"))
		else:
			print("Create TFCG_co and FFCG_co before attempting to create the union: UFCG_co")     
	else:
		#Reading True and False claims csv 
		true_claims=pd.read_csv(os.path.join(rdf_path,"true_claims.csv"),index_col=0)
		false_claims=pd.read_csv(os.path.join(rdf_path,"false_claims.csv"),index_col=0)
		trueclaim_entities,trueclaim_edges,falseclaim_entities,falseclaim_edges={}
		#DBPedia regex to match DBPedia Entities
		entity_regex=re.compile(r'http:\/\/dbpedia\.org\/resource\/')
		#Initializing the two graphs
		TFCG_co=nx.Graph()
		FFCG_co=nx.Graph()
		for claimid in eval("{}claims".format(claim_type)):
			claim_entities=set([])
			g=rdflib.Graph()
			filename="Claim{}.rdf".format(claimid)
			try:
				g.parse(os.path.join(rdf_path,"{} Claims".format(claim_type),filename),format='application/rdf+xml')
			except:
				pass
			for triple in g:
				subject,predicate,obj=list(map(str,triple))
				try:
					if entity_regex.search(subject):
						claim_entities.add(subject)
					if entity_regex.search(obj):
						claim_entities.add(obj)
				except KeyError:
					pass
			if claim_type=="true":
				trueclaim_entities[claimid]=list(claim_entities)
				trueclaim_edges[claimid]=list(combinations(trueclaim_entities[claimid],2))
				TFCG_co.add_edges_from(trueclaim_edges[claimid])
			elif claim_type=="false":
				falseclaim_entities[claimid]=list(claim_entities)
				falseclaim_edges[claimid]=list(combinations(falseclaim_entities[claimid],2))
				FFCG_co.add_edges_from(falseclaim_edges[claimid])
		writepath=os.path.join(graph_path,fcg_label)
		os.makedirs(writepath, exist_ok=True)
		nx.write_edgelist(os.path.join(writepath,"{}.edgelist".format(fcg_label)),data=False)

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Create Co-Occurrence Network from claims processed by FRED')
	parser.add_argument('-r','--rdf', metavar='rdf path',type=str,help='Path to the rdf files created by FRED',default="/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/")
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to store the resultant graph')
	parser.add_argument('-f','--fcg', metavar='FactCheckGraph type',type=str,choices=['true','false','union'],help='True False or Union Co-Occur Graph')
	args=parser.parse_args()
	create_co_occur(args.rdf,args.graphpath,args.fcg.lower())