# -*- coding: utf-8 -*-
from rdflib import Graph
import numpy as np
import pdb
import sys
import os
g = Graph()
# Parsing DBPedia dumps
# g.parse(os.path.join('DBPedia Data',"dbpedia_2016-10.nt"), format="nt")
# g.parse(os.path.join('DBPedia Data',"instance_types_en.ttl"), format="turtle")
# g.parse(os.path.join('DBPedia Data',"mappingbased_objects_en.ttl"), format="turtle")
#Saving the parsed graph
# g.serialize(destination=os.path.join('DBPedia Data','dbpedia_graph.nt'), format='nt')
g.parse('/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_graph.nt',format='nt')

FCG_entities=set(list(np.load("intersect_dbpedia_uris.npy")))
triple_list=[]
pair_list=[]
entity_hitlist=[]
empty_list=[]
i=0
#Looping over triples in the graph
for triple in g:
	#splitting them into subject,predicate,object
	triple=list(map(str,triple))
	subject,predicate,obj=triple
	pair=list([subject,obj])
	#Checking if subject and object is in the FCG_entities set
	if subject in FCG_entities and obj in FCG_entities:
		triple_list.append(triple)
		if str(list([subject,obj])) in set(map(str,pair_list)) or str(list([obj,subject])) in set(map(str,pair_list)):
			empty_list.append(list([subject,obj]))
		else:
			pair_list.append(pair)
	i+=1
print(i)
# with open("FCG_entity_triples_dbpedia.csv",'w',encoding='utf-8') as resultFile:
# 	wr = csv.writer(resultFile)
# 	wr.writerow(triple_list)
print("Triple_list:",len(triple_list))
print("Entity_hitlist:",len(entity_hitlist))
print("Pair_list:",len(pair_list))
print("Empty_list:",len(empty_list))
np.save("intersect_entity_triples_dbpedia.npy",triple_list)
np.save("intersect_entity_pairs_dbpedia.npy",pair_list)

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