# -*- coding: utf-8 -*-
from rdflib import Graph
import numpy as np
import pdb
import sys
import os
from IPython.core.debugger import set_trace
g = Graph()
# Parsing DBPedia dumps
# g.parse(os.path.join('DBPedia Data',"dbpedia_2016-10.nt"), format="nt")
# g.parse(os.path.join('DBPedia Data',"instance_types_en.ttl"), format="turtle")
# g.parse(os.path.join('DBPedia Data',"mappingbased_objects_en.ttl"), format="turtle")
#Saving the parsed graph
# g.serialize(destination=os.path.join('DBPedia Data','dbpedia_graph.nt'), format='nt')
g.parse('/gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph Data/DBPedia Data/dbpedia_graph.nt',format='nt')
fcg_label="fred"
FCG_entities=set(list(np.load("Intersect_entities_{}.npy".format(fcg_label))))
triple_list=[]
pair_list=[]
str_pair_list=[]
entity_hitlist=[]
empty_list=[]
i=0
#Looping over triples in the graph
for triple in g:
	#splitting them into subject,predicate,object
	triple=list(map(str,triple))
	subject,predicate,obj=triple
	pair=tuple([subject,obj])
	#Checking if subject and object is in the FCG_entities set
	if subject in FCG_entities and obj in FCG_entities:
		triple_list.append(triple)
		if str(pair) in set(str_pair_list) or str(pair[::-1]) in set(str_pair_list) or subject==obj:
			empty_list.append(pair)
		else:
			pair_list.append(pair)
			str_pair_list=list(map(str,pair_list))
	i+=1
print(i)
# with open("FCG_entity_triples_dbpedia.csv",'w',encoding='utf-8') as resultFile:
# 	wr = csv.writer(resultFile)
# 	wr.writerow(triple_list)
print("Triple_list:",len(triple_list))
print("Entity_hitlist:",len(entity_hitlist))
print("Pair_list:",len(pair_list))
print("Empty_list:",len(empty_list))
np.save("Intersect_true_entityTriples_{}.npy".format(fcg_label),triple_list)
np.save("Intersect_true_entityPairs_{}.npy".format(fcg_label),pair_list)
np.save("Intersect_ignored_entityPairs_{}.npy".format(fcg_label),empty_list)

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