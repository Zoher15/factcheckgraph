import os
import pandas as pd
import networkx as nx
import re
import argparse
import rdflib

def create_co_occur(rdf_path,graph_path,fcg_label):
	fcg_label=fcg_label+"_co"
	if fcg_label=="ufcg_co":
		#Assumes that TFCG_co and FFCG_co exists
		tfcg_path=os.path.join(graph_path,"tfcg_co","tfcg_co.edgelist")
		ffcg_path=os.path.join(graph_path,"ffcg_co","ffcg_co.edgelist")
		if os.path.exists(tfcgpath) and os.path.exists(ffcgpath):
			tfcg=nx.read_edgelist(tfcgpath,comments="@")
			ffcg=nx.read_edgelist(ffcgpath,comments="@")
			ufcg=nx.compose(tfcg,ffcg)
			fcg_path=os.path.join(graph_path,fcg_label)
			os.makedirs(fcg_path, exist_ok=True)
			nx.write_edgelist(ufcg,os.path.join(fcg_path,"{}.edgelist".format(fcg_label)))
			# nx.write_graphml(ufcg,os.path.join(fcg_path,"{}.graphml".format(fcg_label)),prettyprint=True)
		else:
			print("Create tfcg_co and ffcg_co before attempting to create the union: ufcg_co")     
	else:
		claim_types={"tfcg_co":"true","ffcg_co":"false"}
		claim_type=claim_types[fcg_label]
		claim_IDs=np.load(os.path.join(rdf_path,"{}_claimID.npy".format(claim_type)),index_col=0)
		claim_entities,claim_edges={}
		entity_regex=re.compile(r'http:\/\/dbpedia\.org\/resource\/')
		fcg_co=nx.Graph()
		for claim_ID in claim_IDs:
			claim_entities_set=set([])
			claim_g=rdflib.Graph()
			filename="claim{}.rdf".format(str(claim_ID))
			try:
				claim_g.parse(os.path.join(rdf_path,"{}_claims".format(claim_type),filename),format='application/rdf+xml')
			except:
				pass
			for triple in claim_g:
				subject,predicate,obj=list(map(str,triple))
				try:
					if entity_regex.search(subject):
						claim_entities_set.add(subject)
					if entity_regex.search(obj):
						claim_entities_set.add(obj)
				except KeyError:
					pass
			claim_entities[claim_ID]=list(claim_entities)
			claim_edges[claim_ID]=list(combinations(claim_entities[claim_ID],2))
			fcg_co.add_edges_from(claim_edges[claim_ID])
		fcg_path=os.path.join(graph_path,fcg_label)
		os.makedirs(fcg_path, exist_ok=True)
		nx.write_edgelist(os.path.join(fcg_path,"{}.edgelist".format(fcg_label)),data=False)
		# nx.write_graphml(fcg,os.path.join(fcg_path,"{}.graphml".format(fcg_label)),prettyprint=True)

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create co-cccur graph')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/rdf_files/')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg','ffcg','ufcg'],help='True False or Union FactCheckGraph')
	args=parser.parse_args()
	create_co_occur(args.rdfpath,args.graphpath,args.fcgtype)