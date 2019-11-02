import networkx as nx
import numpy as np
import os
import argparse

def create_ntrue_1false(graph_path,rdf_path):
	write_path=os.path.join(graph_path,"ntrue_1false")
	os.makedirs(write_path,exist_ok=True)
	tfcg=nx.read_edgelist(os.path.join(graph_path,"fred","tfcg","tfcg.edgelist"),comments="@")
	tfcg=max(nx.connected_component_subgraphs(tfcg), key=len)
	tfcg_avsp=nx.average_shortest_path_length(tfcg)
	claim_IDs=np.load(os.path.join(rdf_path,"false_claimID.npy"))
	falseclaim_avsp_d=np.asarray([])
	unused_false_claims=np.asarray([])
	notfound_false_claims=np.asarray([])
	for claim_ID in claim_IDs:
		try:
			falseclaim_g=nx.read_edgelist(os.path.join(rdf_path,"false_claims","claim{}.edgelist".format(claim_ID)),comments="@")
		except FileNotFoundError:
			notfound_false_claims.append(claim_ID)
			continue
		ntrue_1false=nx.compose(tfcg,falseclaim_g)
		if nx.is_connected(ntrue_1false) and len(falseclaim_g.edges())>0:
			file_path=os.path.join(write_path,"tfcg_+claim"+claim_ID)
			os.makedirs(file_path,exist_ok=True)
			nx.write_edgelist(ntrue_1false,os.path.join(file_path,"tfcg_+claim{}.edgelist".format(claim_ID)))
			nx.write_graphml(ntrue_1false,os.path.join(file_path,"tfcg_+claim{}.graphml".format(claim_ID)),prettyprint=True)
			falseclaim_avsp_d.append(tfcg_avsp-nx.average_shortest_path_length(ntrue_1false))
		else:
			unused_false_claims.append(claim_ID)
	np.save(os.path.join(write_path,"falseclaim_avsp_d.npy"),falseclaim_avsp_d)
	np.save(os.path.join(write_path,"unused_false_claims.npy"),unused_false_claims)
	np.save(os.path.join(write_path,"notfound_false_claims.npy"),notfound_false_claims)

def create_ntrue_1true(graph_path,rdf_path):
	write_path=os.path.join(graph_path,"ntrue_1true")
	os.makedirs(write_path,exist_ok=True)
	ntrue_1true=nx.read_edgelist(os.path.join(graph_path,"fred","tfcg","tfcg.edgelist"),comments="@")
	claim_IDs=np.load(os.path.join(rdf_path,"true_claimID.npy"))
	trueclaim_avsp_d=np.asarray([])
	unused_true_claims=np.asarray([])
	notfound_true_claims=np.asarray([])
	for claim_ID2 in np.delete(claim_IDs,i):
		try:
			trueclaim_g=nx.read_edgelist(os.path.join(rdf_path,"true_claims","claim{}.edgelist".format(claim_ID2)),comments="@")
	for i,claim_ID in enumerate(claim_IDs):
		ntrue=nx.Graph()
		for claim_ID2 in np.delete(claim_IDs,i):
			try:
				trueclaim_g=nx.read_edgelist(os.path.join(rdf_path,"true_claims","claim{}.edgelist".format(claim_ID2)),comments="@")
			ntrue=nx.compose(ntrue,trueclaim_g)
		ntrue=max(nx.connected_component_subgraphs(ntrue), key=len)
		ntrue_avsp=nx.average_shortest_path_length(ntrue)
		file_path=os.path.join(write_path,"tfcg_-claim"+claim_ID)
		os.makedirs(file_path,exist_ok=True)
		nx.write_edgelist(ntrue,os.path.join(file_path,"tfcg_-claim{}.edgelist".format(claim_ID)))
		nx.write_graphml(ntrue,os.path.join(file_path,"tfcg_-claim{}.graphml".format(claim_ID)),prettyprint=True)
		trueclaim_g=nx.read_edgelist(os.path.join(rdf_path,"true_claims","claim{}.edgelist".format(claim_ID)),comments="@")
		ntrue_1true=nx.compose(ntrue,trueclaim_g)
		if nx.is_connected(ntrue_1true) and len(trueclaim_g.edges())>0:
			trueclaim_avsp_d.append(nx.average_shortest_path_length(ntrue_1true)-ntrue_avsp)
		else:
			discon_true_claims.append(claim_ID)
	np.save(os.path.join(write_path,"trueclaim_avsp_d.npy"),trueclaim_avsp_d)
	np.save(os.path.join(write_path,"unused_true_claims.npy"),discon_true_claims)
	np.save(os.path.join(write_path,"notfound_true_claims.npy"),notfound_true_claims)

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create leave one out graphs')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/rdf_files/')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs/')
	parser.add_argument('-m','--mode', metavar='mode',type=str,choices=['true','false'],help='True or False Mode')
	args=parser.parse_args()
	if args.mode=='false':
		create_ntrue_1false(args.graphpath,args.rdfpath)
	elif args.mode=='true':
		create_ntrue_1true(args.graphpath,args.rdfpath)