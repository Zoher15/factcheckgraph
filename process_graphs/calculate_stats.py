import networkx as nx
import pandas as pd 
import numpy as np
import argparse
import rdflib
import re
import os
import codecs

#Function to calculate stats of interest for a given graph
def calculate_stats(graph_path,graph_class,g_label):
	#Rows of interests
	stats_index=["Number of Nodes","Number of Entities","Number of Edges","Number of Connected Components","Largest Component Nodes","Largest Component Entities",
	"Largest Component Edges","Average Degree","Kappa/Heterogenity Coefficient","Average Clustering Coefficient","Density","Average Shortest Path Length"]
	graph_stats=pd.DataFrame(index=stats_index,columns=[g_label])
	read_path=os.path.join(graph_path,graph_class,g_label,g_label)
	G=nx.read_edgelist(read_path+".edgelist",comments="@",create_using=nx.MultiGraph)
	# G2=nx.read_edgelist(read_path+"1.edgelist",comments="@",create_using=nx.MultiGraph)
	# g1e=set(list(map(lambda x:(sorted([x[0],x[1]])[0],sorted([x[0],x[1]])[1],x[2]['claim_ID']),list(G.edges(data=True)))))
	# g2e=set(list(map(lambda x:(sorted([x[0],x[1]])[0],sorted([x[0],x[1]])[1],x[2]['claim_ID']),list(G2.edges(data=True)))))
	# import pdb
	# pdb.set_trace()
	entity_regex=re.compile(r'^db:|^fu:')
	nodes=G.nodes()
	if graph_class=="kg":
		entities=list(nodes)
	else:
		entities=[node for node in nodes if entity_regex.match(node)]
	degree_list=[val for (node, val) in G.degree()]
	avg_degree=np.average(degree_list)
	degree_square_list=np.asarray(list(map(np.square,degree_list)))
	avg_square_degree=np.average(degree_square_list)
	graph_stats.loc["Number of Nodes",g_label]=len(list(nodes))
	graph_stats.loc["Number of Entities",g_label]=len(list(entities))
	graph_stats.loc["Number of Edges",g_label]=G.number_of_edges()
	graph_stats.loc["Number of Connected Components",g_label]=len(list(nx.connected_components(G)))
	S=[G.subgraph(c).copy() for c in nx.connected_components(G)]
	largest_cc=max(S, key=len)
	largest_cc_entities=[entity for entity in largest_cc.nodes() if entity_regex.match(entity)]
	graph_stats.loc["Largest Component Nodes",g_label]=len(largest_cc.nodes())
	graph_stats.loc["Largest Component Entities",g_label]=len(largest_cc_entities)
	graph_stats.loc["Largest Component Edges",g_label]=len(largest_cc.edges())
	graph_stats.loc["Average Degree",g_label]=avg_degree
	graph_stats.loc["Average Squared Degree",g_label]=avg_square_degree
	kappa=avg_square_degree/np.square(avg_degree)
	graph_stats.loc["Kappa/Heterogenity Coefficient",g_label]=kappa
	# graph_stats.loc["Average Clustering Coefficient",g_label]=nx.average_clustering(G)
	graph_stats.loc["Density",g_label]=nx.density(G)
	# average path length calculation
	# pathlengths=[]
	# for v in nodes:
	# 	spl=dict(nx.single_source_shortest_path_length(G, v))
	# 	for p in spl:
	# 		pathlengths.append(spl[p])
	# avg_shortest_path=sum(pathlengths)/len(pathlengths)
	# graph_stats.loc["Average Shortest Path Length",g_label]=avg_shortest_path
	if graph_stats.loc["Number of Connected Components",g_label]==1:
		graph_stats.loc["Average Shortest Path Length",g_label]=nx.average_shortest_path_length(G)
	write_path=os.path.join(graph_path,graph_class,g_label,"stats")
	os.makedirs(write_path, exist_ok=True)
	write_path=os.path.join(write_path,g_label)
	graph_stats.to_csv(write_path+"_stats.csv")
	# dist={}
	# for p in pathlengths:
	# 	if p in dist:
	# 		dist[p]+=1
	# 	else:
	# 		dist[p]=1
	# with codecs.open(write_path+"_path_lengths.json","w","utf-8") as f:
	# 	f.write(json.dumps(dist,ensure_ascii=False))

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='calculate stats for graphs')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-gc','--graphclass', metavar='graph class',type=str,choices=['fred','fred1','fred2','fred3','co_occur','backbone_df','backbone_dc','largest_ccf','largest_ccc','kg','old_fred'],help='Class of graph to process')
	parser.add_argument('-gt','--graphtype', metavar='graph type',type=str,help='Type of graph like dbpedia,wikidata,tfcg etc')
	args=parser.parse_args()
	calculate_stats(args.graphpath,args.graphclass,args.graphtype)