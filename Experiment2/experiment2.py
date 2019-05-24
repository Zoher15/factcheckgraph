# -*- coding: utf-8 -*-
import rdflib
import os
import numpy as np

def parse_dbpedia():
	g = rdflib.Graph()
	g.parse('/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_graph.nt',format='nt')
	uris=set([])
	uris_dict={}
	edgelist=set([])
	i=0
	#Looping over triples in the graph
	for triple in g:
		#splitting them into subject,predicate,object
		triple=list(map(str,triple))
		subject,predicate,obj=triple
		#if subject and object have already been seen
		if subject in uris and obj in uris:
			subjid=uris_dict[subject]
			objid=uris_dict[obj]
			if tuple([subjid,objid,1]) not in edgelist and tuple([objid,subjid,1]) not in edgelist:
				edgelist.add(tuple([subjid,objid,1]))
		#if only subject has been seen
		elif subject in uris:
			subjid=uris_dict[subject]
			objid=len(uris)
			uris.add(obj)
			uris_dict[obj]=objid
			edgelist.add(tuple([subjid,objid,1]))
		#if only object has been seen before
		elif obj in uris:
			objid=uris_dict[obj]
			subjid=len(uris)
			uris.add(subject)
			uris_dict[subject]=subjid
			edgelist.add(tuple([subjid,objid,1]))
		#if neither have been seen before
		else:
			subjid=len(uris)
			uris.add(subject)
			uris_dict[subject]=subjid
			objid=len(uris)
			uris.add(obj)
			uris_dict[obj]=objid
			edgelist.add(tuple([subjid,objid,1]))
		i+=1
	print(i)
	print(len(edgelist))
	print(len(uris))
	uris=list(uris)
	edgelist=list(edgelist)
	edgelist=np.asarray([list(i) for i in edgelist])
	np.save("dbpedia_uris.npy",uris)
	np.save("dbpedia_edgelist.npy",edgelist)
	with codecs.open("dbpedia_uris.txt","w","utf-8") as f:
		for uri in uris:
			try:
				f.write(str(uri)+"\n")
			except:
				pdb.set_trace()
	with codecs.open("dbpedia_edgelist.txt","w","utf-8") as f:
		for line in edgelist:
			f.write("{} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2])))
	return uris_dict,edgelist

uris_dict,edgelist=parse_dbpedia()