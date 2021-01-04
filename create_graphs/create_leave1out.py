import os
import pandas as pd
import multiprocessing as mp
import numpy as np
import networkx as nx
import sys
import argparse

def compose_graphs(claims_path,fcg_path,suffix,claim_IDs,claim_IDs2remove,graph_type):
	for skipID in claim_IDs2remove:
		temp_claim_IDs=claim_IDs.copy()
		temp_claim_IDs.remove(skipID)
		fcg=eval(graph_type+"()")
		pathname=fcg_path+"-"+str(skipID)+".edgelist"
		edgelist=[]
		# if not os.path.isfile(pathname): 
		for claim_ID in temp_claim_IDs:
			filename=os.path.join(claims_path,"claim{}".format(str(claim_ID))+"_"+suffix)
			try:
				claim_g=nx.read_edgelist(filename+".edgelist",comments="@")
			except:
				continue
			edgelist+=list(claim_g.edges.data())
		fcg.add_edges_from(sorted(edgelist,key=lambda x:x[0]))
		nx.write_edgelist(fcg,pathname)

def create_leave1out(rdf_path,graph_path,fcg_class,fcg_label,cpu,jobs,jobnum,graph_type):
	claim_types={"tfcg_co":"true","ffcg_co":"false","tfcg":"true","ffcg":"false"}
	claim_type=claim_types[fcg_label]
	claim_IDs=list(np.load(os.path.join(rdf_path,"{}_claimID.npy".format(claim_type))))
	suffix={"tfcg_co":"co","ffcg_co":"co","tfcg":"clean","ffcg":"clean"}
	fcg_path=os.path.join(graph_path,fcg_class,fcg_label,"leave1out")
	os.makedirs(fcg_path,exist_ok=True)
	claims_path=os.path.join(rdf_path,"{}_claims".format(claim_type))
	n=int(len(claim_IDs)/(cpu*jobs))+1
	start=cpu*(jobnum-1)
	end=cpu*(jobnum)
	if cpu>1:
		pool=mp.Pool(processes=cpu)
		results=[pool.apply_async(compose_graphs, args=(claims_path,os.path.join(fcg_path,fcg_label),suffix[fcg_label],claim_IDs,claim_IDs[index*n:(index+1)*n],graph_type)) for index in range(start,end)]
		output=[p.get() for p in results]
	else:
		pool=mp.Pool(processes=cpu)
		compose_graphs(claims_path,os.path.join(fcg_path,fcg_label),suffix[fcg_label],claim_IDs,claim_IDs[0:n],graph_type)

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create fred graph')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/rdf_files/')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-fc','--fcgclass', metavar='FactCheckGraph Class',type=str,choices=['fred','co_occur'],help='Class of factcheckgraph to process')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,help='True False or Union FactCheckGraph')
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	parser.add_argument('-jobs','--jobs',metavar='Number of Jobs',type=int,help='Number of Jobs task is divided into',default=1)
	parser.add_argument('-jn','--jobnum',metavar='Number of Jobs',type=int,help='Number of Job executing',default=1)
	parser.add_argument('-gt','--graphtype', metavar='Graph Type Directed/Undirected',type=str,choices=['directed','undirected'],default='undirected')
	args=parser.parse_args()
	graph_types={'undirected':'nx.MultiGraph','directed':'nx.MultiDiGraph'}
	create_leave1out(args.rdfpath,args.graphpath,args.fcgclass,args.fcgtype,args.cpu,args.jobs,args.jobnum,graph_types[args.graphtype])