import networkx as nx
import pandas as pd 
import numpy as np
import argparse
import rdflib
import re
import os
import codecs

#Function to calculate stats of interest for a given graph
def calculate_stats(graph_path,graph_class,graph_type):
	g_labels={"fred":{"true":"TFCG","false":"FFCG","union":"UFCG"},
	"co-occur":{"true":"TFCG_co","false":"FFCG_co","union":"UFCG_co"},
	"backbone":{"true":"TFCG_bb","false":"FFCG_bb","union":"UFCG_bb"},
	"kg":{"dbpedia":"DBPedia","wikidata":"Wikidata"}}
	g_label=fcg_modes[graph_class][graph_type]
	degreelist={}
	pathlengths={}
	stats_index=["Number of Nodes","Number of Entities","Number of Edges","Number of Connected Components","Largest Component Nodes","Largest Component Entities",
	"Largest Component Edges","Average Degree","Kappa/Heterogenity Coefficient","Average Clustering Coefficient","Density","Average Shortest Path Length"]
	graph_stats=pd.DataFrame(index=stats_index,columns=g_labels)
	readpath=os.path.join(graph_path,graph_class,g_label)
	if graph_class=="kg":
		try:
			#Trying if KG exists in edgelist form
			G=nx.read_edgelist(os.path.join(readpath,"{}.edgelist".format(g_label)))
		except FileNotFoundError:
			#Reading KG from .nt format and saving in edgelist form
			g = rdflib.Graph()
			g.parse(os.path.join(readpath,'{}.nt'.format(g_label)),format='nt')
			G=nx.Graph()
			for (a,b,c) in g.getEdges():
				G.add_edge(a,c,label=b)
			G.write_edgelist(os.path.join(readpath,'{}.edgelist'.format(g_label)),data=True)
			nodes=G.nodes()
			entities=nodes
	else:
		G=nx.read_edgelist(os.path.join(readpath,"{}.edgelist".format(g_label)))
		entity_regex=re.compile(r'http:\/\/dbpedia\.org\/resource\/')
		nodes=G.nodes()
		entities=[node for node in nodes if entity_regex.match(node)]
	G=nx.read_edgelist(os.path.join(readpath,"{}.edgelist".format(g_label)))
	degree_list=[val for (node, val) in G.degree()]
	avg_degree=np.average(degreelist)
	degree_square_list=np.asarray(list(map(np.square,degree_list)))
	avg_square_degree=np.average(degree_square_list)
	writepath=os.path.join(readpath,"stats")
	os.makedirs(writepath, exist_ok=True)
	graph_stats.iloc["Number of Nodes",g_label]=len(list(nodes))
	graph_stats.iloc["Number of Entities",g_label]=len(list(entities))
	graph_stats.iloc["Number of Edges",g_label]=G.number_of_edges()
	graph_stats.iloc["Number of Connected Components",g_label]=len(list(nx.connected_components(G)))
	largest_cc=max(nx.connected_component_subgraphs(G), key=len)
	largest_cc_entities=[entity for entity in largest_cc.nodes() if entity_regex.match(entity)]
	graph_stats.iloc["Largest Component Nodes",g_label]=len(largest_cc.nodes())
	graph_stats.iloc["Largest Component Entities",g_label]=len(largest_cc_entities)
	graph_stats.iloc["Largest Component Edges",g_label]=len(largest_cc.edges())
	graph_stats.iloc["Average Degree",g_label]=avg_degree
	graph_stats.iloc["Average Squared Degree",g_label]=avg_square_degree
	graph_stats.iloc["Kappa/Heterogenity Coefficient",g_label]=kappa
	graph_stats.iloc["Average Clustering Coefficient",g_label]=nx.average_clustering(G)
	graph_stats.iloc["Density",g_label]=nx.density(G)
	#average path length calculation
	pathlengths=[]
	for v in nodes:
		spl=dict(nx.single_source_shortest_path_length(G, v))
		for p in spl:
			pathlengths.append(spl[p])
	avg_shortest_path=sum(pathlengths)/len(pathlengths)
	graph_stats.iloc["Average Shortest Path Length",g_label]=avg_shortest_path
	graph_stats.to_csv(os.path.join(writepath,"{}_stats.csv".format(g_label)))
	dist={}
	for p in pathlengths:
		if p in dist:
			dist[p]+=1
		else:
			dist[p]=1
	with codecs.open(os.path.join(writepath,"{}_path_lengths.json".format(g_label)),"w","utf-8") as f:
		f.write(json.dumps(dist,ensure_ascii=False))

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='calculate stats for graphs')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory')
	parser.add_argument('-gc','--graphclass', metavar='graph class',type=str,choices=['fred','co-occur','backbone','kg'],help='Class of graph to process')
	parser.add_argument('-gt','--graphtype', metavar='graph type',type=str,choices=['true','false','union','dbpedia','wikidata'],help='True, False, Union, DBPedia or Wikidata Graph')
	args=parser.parse_args()
	calculate_stats(args.graphpath,args.graphclass,args.graphtype)
