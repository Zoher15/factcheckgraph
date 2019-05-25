# -*- coding: utf-8 -*-
import rdflib
import os
import numpy as np
import json
import pandas as pd
import re
import pdb
import codecs
#Function to parse dbpedia, get uris create an ID dictionary and save it in the form of edgelist
#This format is to enable use of Knowledge Linker. Hence a uris.txt file is created for index 
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
	np.save("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_uris.npy",uris)
	np.save("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_edgelist.npy",edgelist)
	with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_uris_dict.json","w","utf-8") as f:
		f.write(json.dumps(uris_dict))
	with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_uris.txt","w","utf-8") as f:
		for uri in uris_dict.keys():
			try:
				f.write(str(uri)+"\n")
			except:
				pdb.set_trace()
	with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_edgelist.txt","w","utf-8") as f:
		for line in edgelist:
			f.write("{} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2])))
	return uris,uris_dict,edgelist

def load_stuff():
	uris=np.load("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_uris.npy")
	with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_uris_dict.json","r","utf-8") as f:
		uris_dict=json.loads(f.read())
	# edgelist=np.load("/gpfs/home/z/k/zkachwal/Carbonate/DBPedia Data/dbpedia_edgelist.npy")
	return uris,uris_dict#,edgelist


def parse_claims(uris_dict):
	data=pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/claimreviews_db2.csv",index_col=0)
	##Dropping non-str rows
	filter=list(map(lambda x:type(x)!=str,data['rating_name']))
	data.drop(data[filter].index,inplace=True)
	print(data.groupby('fact_checkerID').count())
	trueregex=re.compile(r'(?i)^true|^correct$|^mostly true$|^geppetto checkmark$')
	falseregex=re.compile(r'(?i)^false|^mostly false|^pants on fire$|^four pinocchios$|^no\ |^no:|^distorts the facts|^wrong$')
	trueind=data['rating_name'].apply(lambda x:trueregex.match(x)!=None)
	trueclaims=list(data.loc[trueind]['claimID'])
	falseind=data['rating_name'].apply(lambda x:falseregex.match(x)!=None)
	falseclaims=list(data.loc[falseind]['claimID'])
	trueclaim_uris=[]
	falseclaim_uris=[]
	dbpediaregex=re.compile(r'http:\/\/dbpedia\.org\/resource\/')
	np.save("true_claimID_list.npy",trueclaims)
	np.save("false_claimID_list.npy",falseclaims)
	for t in trueclaims[:20]:
		claim_uris=set([])
		g=rdflib.Graph()
		filename="claim"+str(t)+".rdf"
		try:
			g.parse("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/"+filename,format='application/rdf+xml')
		except:
			# continue
			pass
		for triple in g:
			subject,predicate,obj=list(map(str,triple))
			try:
				if dbpediaregex.search(subject):
					claim_uris.add(uris_dict[subject])
				if dbpediaregex.search(obj):
					claim_uris.add(uris_dict[obj])
			except KeyError:
				continue
		trueclaim_uris.append(list(claim_uris))
	for f in falseclaims[:20]:
		claim_uris=set([])
		g=rdflib.Graph()
		filename="claim"+str(f)+".rdf"
		try:
			g.parse("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/"+filename,format='application/rdf+xml')
		except:
			# continue
			pass
		for triple in g:
			subject,predicate,obj=list(map(str,triple))
			try:
				if dbpediaregex.search(subject):
					claim_uris.add(uris_dict[subject])
				if dbpediaregex.search(obj):
					claim_uris.add(uris_dict[obj])
			except KeyError:
				continue
		falseclaim_uris.append(list(claim_uris))
	np.save("trueclaim_uris.npy",trueclaim_uris)
	np.save("falseclaim_uris.npy",falseclaim_uris)

# uris,uris_dict,edgelist=parse_dbpedia()
uris,uris_dict=load_stuff()
parse_claims(uris_dict)