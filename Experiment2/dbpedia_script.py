# -*- coding: utf-8 -*-
from rdflib import Graph
import numpy as np
import pdb
import sys
import os
import networkx as nx 
from decimal import Decimal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# g = Graph()
# # Parsing DBPedia dumps
# # g.parse(os.path.join('DBPedia Data',"dbpedia_2016-10.nt"), format="nt")
# # g.parse(os.path.join('DBPedia Data',"instance_types_en.ttl"), format="turtle")
# # g.parse(os.path.join('DBPedia Data',"mappingbased_objects_en.ttl"), format="turtle")
# #Saving the parsed graph
# # g.serialize(destination=os.path.join('DBPedia Data','dbpedia_graph.nt'), format='nt')
# g.parse('/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph Data/DBPedia Data/dbpedia_graph.nt',format='nt')

# FCG_entities=set(list(np.load("intersect_dbpedia_uris_co.npy")))
# triple_list=[]
# pair_list=[]
# entity_hitlist=[]
# empty_list=[]
# i=0
# #Looping over triples in the graph
# for triple in g:
# 	#splitting them into subject,predicate,object
# 	triple=list(map(str,triple))
# 	subject,predicate,obj=triple
# 	pair=list([subject,obj])
# 	#Checking if subject and object is in the FCG_entities set
# 	if subject in FCG_entities and obj in FCG_entities:
# 		triple_list.append(triple)
# 		if str(list([subject,obj])) in set(map(str,pair_list)) or str(list([obj,subject])) in set(map(str,pair_list)):
# 			empty_list.append(list([subject,obj]))
# 		else:
# 			pair_list.append(pair)
# 	i+=1
# print(i)
# # with open("FCG_entity_triples_dbpedia.csv",'w',encoding='utf-8') as resultFile:
# # 	wr = csv.writer(resultFile)
# # 	wr.writerow(triple_list)
# print("Triple_list:",len(triple_list))
# print("Entity_hitlist:",len(entity_hitlist))
# print("Pair_list:",len(pair_list))
# print("Empty_list:",len(empty_list))
# np.save("intersect_entity_triples_dbpedia_co.npy",triple_list)
# np.save("intersect_entity_true_pairs_dbpedia_co.npy",pair_list)

# for mode in ["FFCG","TFCG"]:
# 	# FCG_entities=set(np.load(os.path.join(mode,mode+"_dbpedia_uris.npy")))
# 	FCG_entities=set(np.load(os.path.join(mode,mode+"_dbpedia_uris.npy")))
# 	triple_list=[]
# 	entity_hitlist=[]
# 	i=0
# 	#Looping over triples in the graph
# 	for triple in g:
# 		#splitting them into subject,predicate,object
# 		triple=list(map(str,triple))
# 		subject,predicate,obj=triple
# 		#Checking if subject and object is in the FCG_entities set
# 		if subject in FCG_entities and obj in FCG_entities:
# 			triple_list.append(triple)
# 		# #Checking if subject is in the FCG_entities set
# 		# elif subject in FCG_entities:
# 		# 	entity_hitlist.append(subject)
# 		# #Checking if object is in the FCG_entities set
# 		# elif obj in FCG_entities:
# 		# 	entity_hitlist.append(obj)
# 		i+=1
# 		# np.save("entity_hitlist.npy",entity_hitlist)
# 	# np.save("dbpedia_entities.npy",dbpedia_entities)
# 	print(i)
# 	# with open("FCG_entity_triples_dbpedia.csv",'w',encoding='utf-8') as resultFile:
# 	# 	wr = csv.writer(resultFile)
# 	# 	wr.writerow(triple_list)
# 	print(len(triple_list))
# 	print(len(entity_hitlist))
# 	np.save(os.path.join(mode,mode+"_entity_triples_dbpedia.npy"),triple_list)

def calculate_DBPedia_stats():
	with open("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph Data/DBPedia Data/DBPedia_stats.txt","w") as f:
		try:
			G=nx.read_weighted_edgelist("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph Data/DBPedia Data/dbpedia_edgelist.txt")
		except:
			pdb.set_trace()
		degreelist=list(G.degree())
		degreelist=list(map(lambda x:x[1],degreelist))
		f.write("Number of DBpedia Uris: %.2E \n" % (len(G)))
		degreefreq=np.asarray([float(0) for i in range(max(degreelist)+1)])
		for degree in degreelist:
			degreefreq[degree]+=1
		degreeprob=degreefreq/sum(degreefreq)
		plt.figure()	
		plt.loglog(range(0,max(degreelist)+1),degreeprob)
		plt.xlabel('Degree')
		plt.ylabel('Probability')
		plt.title('Degree Distribution')
		plt.savefig("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph Data/DBPedia Data/dbpedia_degreedist.png")
		degree_square_list=np.asarray(list(map(np.square,degreelist)))
		f.write("Number of Edges: %.2E \n" %Decimal(G.number_of_edges()))
		f.write("Number of Nodes: %.2E \n" %Decimal(G.number_of_nodes()))
		f.write("Number of Connected Components: %s \n" %(len(list(nx.connected_components(G)))))
		largest_cc = max(nx.connected_component_subgraphs(G), key=len)
		f.write("Largest Component Edges: %s \n" %(len(largest_cc.edges())))			
		f.write("Largest Component Nodes: %s \n" %(len(largest_cc.nodes())))
		f.write("Average Degree: %.2E \n" %Decimal(np.average(degreelist)))
		f.write("Average Squared Degree: %.2E \n" %Decimal(np.average(degree_square_list)))
		kappa=np.average(degree_square_list)/(np.square(np.average(degreelist)))
		f.write("Kappa/Heterogenity Coefficient (average of squared degree/square of average degree): %.2E \n" % (kappa))
		f.write("Average Clustering Coefficient: %.2E \n" % (nx.average_clustering(G)))
		f.write("Density: %.2E \n" %(nx.density(G)))
		#average path length calculation
		pathlengths = []
		for v in G.nodes():
			spl = dict(nx.single_source_shortest_path_length(G, v))
			for p in spl:
				pathlengths.append(spl[p])
		f.write("Average Shortest Path Length: %.2E \n\n" %Decimal(sum(pathlengths) / len(pathlengths)))
		dist = {}
		for p in pathlengths:
			if p in dist:
				dist[p] += 1
			else:
				dist[p] = 1
		f.write("Length #Paths \n")
		verts = dist.keys()
		pathlen=[]
		for d in sorted(verts):
			f.write('%s %d \n' % (d, dist[d]))
			pathlen.append([d,dist[d]])
		pathlen=np.asarray(pathlen)
		x=pathlen[:,0]
		y=pathlen[:,1]
		plt.figure()	
		plt.bar(x,y)
		plt.xlabel('Path Length')
		plt.ylabel('Number of Times')
		plt.title("DBPedia Distribution of Path Lengths")
		plt.savefig("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph Data/DBPedia Data/dbpedia_pathlen.png")
		np.save("/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph Data/DBPedia Data/dbpedia_pathlen.npy",pathlen)
calculate_DBPedia_stats()