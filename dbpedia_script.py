# -*- coding: utf-8 -*-
from rdflib import Graph
import numpy as np
import pdb
import sys
import os
g = Graph()
# Parsing DBPedia dumps
# g.parse("dbpedia_2016-10.nt", format="nt")
# g.parse("instance_types_en.ttl", format="turtle")
# g.parse("mappingbased_objects_en.ttl", format="turtle")
#Saving the parsed graph
# g.serialize(destination='dbpedia_graph.nt', format='nt')
# g.parse('test.ttl',format='turtle')
#$$$$$$$$$$$$$$$$$CSV Mode
#Loading the list of entities from FCG
# import csv
# a=[]
# with open("dbpedia_entities_FCG.csv",encoding='utf-8') as csvfile:
# 	read=csv.reader(csvfile)
# 	for row in read:
# 		a+=row
# FCG_entities=set(a)
g.parse('dbpedia_graph.nt',format='nt')
for mode in ["FFCG","TFCG"]:
	FCG_entities=set(np.load(os.path.join(mode,mode+"_dbpedia_uris.npy")))
	triple_list=[]
	entity_hitlist=[]
	i=0
	#Looping over triples in the graph
	for triple in g:
		#splitting them into subject,predicate,object
		triple=list(map(str,triple))
		subject,predicate,obj=triple
		#Checking if subject and object is in the FCG_entities set
		if subject in FCG_entities and obj in FCG_entities:
			triple_list.append(triple)
		# #Checking if subject is in the FCG_entities set
		# elif subject in FCG_entities:
		# 	entity_hitlist.append(subject)
		# #Checking if object is in the FCG_entities set
		# elif obj in FCG_entities:
		# 	entity_hitlist.append(obj)
		i+=1
		# np.save("entity_hitlist.npy",entity_hitlist)
	# np.save("dbpedia_entities.npy",dbpedia_entities)
	print(i)
	# with open("FCG_entity_triples_dbpedia.csv",'w',encoding='utf-8') as resultFile:
	# 	wr = csv.writer(resultFile)
	# 	wr.writerow(triple_list)
	print(len(triple_list))
	print(len(entity_hitlist))
	np.save(os.path.join(mode,mode+"_entity_triples_dbpedia.npy"),triple_list)
	# np.save("FCG_entities_after.npy",list(FCG_entities))