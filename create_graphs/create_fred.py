import os
import re
import sys
import time
import rdflib
import requests
import argparse
import networkx as nx
from flufl.enum import Enum
from rdflib import plugin
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import codecs
from rdflib.serializer import Serializer
from rdflib.plugins.memory import IOMemory
from IPython.core.debugger import set_trace
import json
import numpy as np
import xml.sax
import html
from collections import ChainMap 
from itertools import chain
import multiprocessing as mp
from urllib.parse import urlparse
from pprint import pprint
from flatten_dict import flatten as flatten_dict

# Original script edited by Zoher Kachwala for FactCheckGraph

class FredGraph:
	def __init__(self,rdf):
		self.rdf=rdf

	def getEdges(self):
		return [(a.strip(),b.strip(),c.strip()) for (a,b,c) in self.rdf]

def preprocessText(text):
	nt=text.replace("-"," ")
	nt=nt.replace("#"," ")
	nt=nt.replace(chr(96),"'") #`->'
	nt=nt.replace("'nt "," not ")
	nt=nt.replace("'ve "," have ")
	nt=nt.replace(" what's "," what is ")
	nt=nt.replace("What's ","What is ")
	nt=nt.replace(" where's "," where is ")
	nt=nt.replace("Where's ","Where is ")
	nt=nt.replace(" how's "," how is ")
	nt=nt.replace("How's ","How is ")
	nt=nt.replace(" he's "," he is ")
	nt=nt.replace(" she's "," she is ")
	nt=nt.replace(" it's "," it is ")
	nt=nt.replace("He's ","He is ")
	nt=nt.replace("She's ","She is ")
	nt=nt.replace("It's ","It is ")
	nt=nt.replace("'d "," had ")
	nt=nt.replace("'ll "," will ")
	nt=nt.replace("'m "," am ")
	nt=nt.replace(" ma'am "," madam ")
	nt=nt.replace(" o'clock "," of the clock ")
	nt=nt.replace(" 're "," are ")
	nt=nt.replace(" y'all "," you all ")

	nt=nt.strip()
	if nt[len(nt)-1]!='.':
		nt=nt + "."

	return nt

def getFredGraph(sentence,key,filename):
	url="http://wit.istc.cnr.it/stlab-tools/fred"
	header={"Accept":"application/rdf+xml","Authorization":key}
	data={'text':sentence,'semantic-subgraph':True}
	r=requests.get(url,params=data,headers=header)
	with open(filename, "w") as f:
		f.write("<?xml version='1.0' encoding='UTF-8'?>\n"+r.text)
	# return openFredGraph(filename),r
	return r

def openFredGraph(filename):
	rdf=rdflib.Graph()
	rdf.parse(filename)
	return FredGraph(rdf)

def label_mapper(x,text):
	try:
		d={
		bool(re.match('url',text)):'review_url',
		bool(re.match('datePublished',text)):'review_date',
		bool(re.match('dateModified',text)):'review_modified_date',
		bool(re.match('image.(\w+\.)?url',text)):'review_img',
		bool(re.match('headline',text)):'review_headline',
		bool(re.match('author.(\w+\.)?name',text)):'review_author_name',
		bool(re.match('reviewRating.(\w+\.)?ratingValue',text)):'rating_value',
		bool(re.match('reviewRating.(\w+\.)?bestRating',text)):'best_rating',
		bool(re.match('reviewRating.(\w+\.)?worstRating',text)):'worstRating',
		bool(re.match('reviewRating.(\w+\.)?alternateName',text)):'rating_name',
		bool(re.match('reviewRating.(\w+\.)?image',text)):'review_rating_img',
		bool(re.match('author.name',text)):'fact_checker_name',
		bool(re.match('author.(\w+\.)?image',text)):'fact_checker_img',
		bool(re.match('author.(\w+\.)?url',text)):'fact_checker_url',
		bool(re.match('description',text)):'claim_description',
		bool(re.match('claimReviewed',text)):'claim_text',
		bool(re.match('itemReviewed.(\w+\.)?datePublished',text)):'claim_date',
		bool(re.match('itemReviewed.(\w+\.)?author.(\w+\.)?jobTitle',text)):'claim_author_job',
		bool(re.match('itemReviewed.(\w+\.)?author.(\w+\.)?name',text)):'claim_author_name',
		bool(re.match('itemReviewed.(\w+\.)?image',text)):'claim_author_img',
		bool(re.match('itemReviewed.(\w+\.)?name',text)):'claim_location',
		bool(re.match('(\w+\.)*sameAs',text)):'claim_location_url'
		}
		return d[True]
	except KeyError:
		return

#function to check a claim and create dictionary of edges to contract and remove
def checkClaimGraph(g,mode):
	if mode=='rdf':
		claim_g=nx.Graph()
	elif mode=='nx':
		claim_g=g
	regex_27=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#%27.*')
	regex_det=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/quantifiers\.owl.*$')
	regex_data=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/dul\/DUL\.owl#hasDataValue$')
	regex_event=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/dul\/DUL\.owl#Event.*')
	regex_prop=re.compile(r'^http:\/\/www\.w3\.org\/2002\/07\/owl#DataTypeProperty$')
	regex_sameas=re.compile(r'^http:\/\/www\.w3\.org\/2002\/07\/owl#sameAs$')
	regex_equiv=re.compile(r'^http:\/\/www\.w3\.org\/2002\/07\/owl#equivalentClass$')
	regex_sub=re.compile(r'^http:\/\/www\.w3\.org\/2000\/01\/rdf-schema#subClassOf$')
	regex_assoc=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/dul\/DUL\.owl#associatedWith$')
	regex_type=re.compile(r'^http:\/\/www\.w3\.org\/1999\/02\/22-rdf-syntax-ns#type$')
	regex_fred=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#([a-zA-Z]*)_.*')
	regex_fredup=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#([A-Z]+[a-zA-Z]*)')
	regex_vn=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/vn\/data\/([a-zA-Z]*)_.*')
	regex_dbpedia=re.compile(r'^http:\/\/dbpedia\.org\/resource\/(.*)')
	regex_quant=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/quantifiers\.owl#.*')
	nodes2contract={}
	nodes2remove={}
	nodes2remove['%27']=[]
	nodes2remove['det']=[]
	nodes2remove['url']=[]
	nodes2remove['prop']=[]
	nodes2remove['data']=[]
	nodes2contract['type']=[]
	nodes2contract['sub']=[]
	nodes2contract['vnequiv']=[]
	nodes2contract['dbpediasas']=[]
	nodes2contract['dbpediaequiv']=[]
	nodes2contract['dbpediassoc']=[]
	if mode=='rdf':
		edge_list=g.getEdges()
	elif mode=='nx':
		edge_list=claim_g.edges.data('label', default='')
	for (a,c,b) in edge_list:
		if mode=='rdf':
			t=c 
			c=b 
			b=t
			claim_g.add_edge(a,c,label=b)
		a_urlparse=urlparse(a)
		c_urlparse=urlparse(c)
		#Delete '%27' edges
		if regex_27.match(a):
			nodes2remove['%27'].append(a)
		elif regex_27.match(c):
			nodes2remove['%27'].append(c)
		#Merging verbs like show_1 and show_2 with show
		if regex_type.match(b):
			if (regex_fredup.match(a) and not regex_fred.match(c)) and (regex_fredup.match(a)[1].split("_")[0].lower() in c.split("\\")[-1].lower()):
				nodes2contract['type'].append((c,a))
			elif (regex_fredup.match(c) and not regex_fred.match(a)) and (regex_fredup.match(c)[1].split("_")[0].lower() in a.split("\\")[-1].lower()):
				nodes2contract['type'].append((a,c))
			elif regex_fred.match(a) and (regex_fred.match(a)[1].lower() in c.split("\\")[-1].lower()):
				nodes2contract['type'].append((c,a))
			elif (regex_fred.match(c)) and (regex_fred.match(c)[1].lower() in a.split("\\")[-1].lower()):
				nodes2contract['type'].append((a,c))
		#Merging verbs like show_1 and show_2 with show for subclass predicates
		elif regex_sub.match(b):
			if (regex_fredup.match(a) and not regex_fred.match(c)) and (regex_fredup.match(a)[1].split("_")[0].lower() in c.split("\\")[-1].lower()):
				nodes2contract['sub'].append((c,a))
			elif (regex_fredup.match(c) and not regex_fred.match(a)) and (regex_fredup.match(c)[1].split("_")[0].lower() in a.split("\\")[-1].lower()):
				nodes2contract['sub'].append((a,c))
			elif regex_fred.match(a) and (regex_fred.match(a)[1].lower() in c.split("\\")[-1].lower()):
				nodes2contract['sub'].append((c,a))
			elif (regex_fred.match(c)) and (regex_fred.match(c)[1].lower() in a.split("\\")[-1].lower()):
				nodes2contract['sub'].append((a,c))
		#Merging verbs with their verbnet forms
		elif regex_equiv.match(b) and (regex_vn.match(a) or regex_vn.match(c)):
			if (regex_fredup.match(a) and regex_vn.match(c)) and (regex_fredup.match(a)[1].split("_")[0].lower() in regex_vn.match(c)[1].lower()):
				nodes2contract['vnequiv'].append((c,a))
			elif (regex_fredup.match(c) and regex_vn.match(a)) and (regex_fredup.match(c)[1].split("_")[0].lower() in regex_vn.match(a)[1].lower()):
				nodes2contract['vnequiv'].append((a,c))
		#Merging nodes with sameAs relationships
		elif regex_sameas.match(b) and (regex_dbpedia.match(a) or regex_dbpedia.match(c)):
			if (regex_fredup.match(a) and regex_dbpedia.match(c)) and (regex_fredup.match(a)[1]!="Of" and regex_fredup.match(a)[1]!="Thing"):
				nodes2contract['dbpediasas'].append((c,a))
			elif (regex_fredup.match(c) and regex_dbpedia.match(a)) and (regex_fredup.match(c)[1]!="Of" and regex_fredup.match(c)[1]!="Thing"):
				nodes2contract['dbpediasas'].append((a,c))
		#Merging nodes with equivalentClass relationships
		elif regex_equiv.match(b) and (regex_dbpedia.match(a) or regex_dbpedia.match(c)):
			if (regex_fredup.match(a) and regex_dbpedia.match(c)) and (regex_fredup.match(a)[1]!="Of" and regex_fredup.match(a)[1]!="Thing"):
				nodes2contract['dbpediaequiv'].append((c,a))
			elif (regex_fredup.match(c) and regex_dbpedia.match(a)) and (regex_fredup.match(c)[1]!="Of" and regex_fredup.match(c)[1]!="Thing"):
				nodes2contract['dbpediaequiv'].append((a,c))
		#Merging nodes with associatedWith relationships
		elif regex_assoc.match(b) and (regex_dbpedia.match(a) or regex_dbpedia.match(c)):
			if (regex_fredup.match(a) and regex_dbpedia.match(c)) and (regex_fredup.match(a)[1]!="Of" and regex_fredup.match(a)[1]!="Thing") and (regex_fredup.match(a)[1].split("_")[0].lower() in regex_dbpedia.match(c)[1].lower()):
				nodes2contract['dbpediassoc'].append((c,a))
			elif (regex_fredup.match(c) and regex_dbpedia.match(a)) and (regex_fredup.match(c)[1]!="Of" and regex_fredup.match(c)[1]!="Thing") and (regex_fredup.match(c)[1].split("_")[0].lower() in regex_dbpedia.match(a)[1].lower()):
				nodes2contract['dbpediassoc'].append((a,c))
		#remove hasDeterminer 'The'
		elif regex_det.match(b):
			if regex_quant.match(a):
				nodes2remove['det'].append(a)
			elif regex_quant.match(c):
				nodes2remove['det'].append(c)
		elif regex_data.match(a) or regex_event.match(a):
			nodes2remove['data'].append(a)
		elif regex_data.match(c) or regex_event.match(c):
			nodes2remove['data'].append(c)
		elif regex_prop.match(a):
			nodes2remove['prop'].append(a)
		elif regex_prop.match(c):
			nodes2remove['prop'].append(a)
		elif (a_urlparse.netloc=='' and a_urlparse.scheme==''):
			nodes2remove['url'].append(a)
		elif (c_urlparse.netloc=='' and c_urlparse.scheme==''):
			nodes2remove['url'].append(c)
	return claim_g,nodes2remove,nodes2contract

#fetch fred graph files from their API. slow and dependent on rate
def fredParse(claims_path,claims,init,end):
	key="Bearer 0d9d562e-a2aa-30df-90df-d52674f2e1f0"
	errorclaimid=[]
	#fred starts
	start=time.time()
	start2=time.time()
	daysec=86400
	minsec=60
	fcg=nx.Graph()
	rdf=rdflib.Graph()
	clean_claims={}
	for i in range(init,end):
		dif=abs(time.time()-start)
		diff=abs(daysec-dif)
		while True:
			try:
				dif=abs(time.time()-start)
				dif2=abs(time.time()-start2)
				diff=abs(daysec-dif)
				claim_ID=claims.iloc[i]['claimID']
				sentence=html.unescape(claims.iloc[i]['claim_text']).replace("`","'")
				print("Index:",i,"Claim ID:",claim_ID," DayLim2Go:",round(diff),"MinLim2Go:",round(min(abs(minsec-dif2),60)))
				filename=os.path.join(claims_path,"claim{}".format(str(claim_ID)))
				r=getFredGraph(preprocessText(sentence),key,filename+".rdf")
				if "You have exceeded your quota" not in r.text and "Runtime Error" not in r.text and "Service Unavailable" not in r.text:
					if r.status_code in range(100,500) and r.text:
						g=openFredGraph(filename+".rdf")
						claim_g,nodes2remove,nodes2contract=checkClaimGraph(g,'rdf')
						#store pruning data
						clean_claims[str(claim_ID)]={}
						clean_claims[str(claim_ID)]['nodes2remove']=nodes2remove
						clean_claims[str(claim_ID)]['nodes2contract']=nodes2contract
						#write pruning data
						with codecs.open(filename+"_clean.json","w","utf-8") as f:
							f.write(json.dumps(clean_claims[str(claim_ID)],indent=4,ensure_ascii=False))
						#write claim graph as edgelist and graphml
						nx.write_edgelist(claim_g,filename+".edgelist")
						claim_g=nx.read_edgelist(filename+".edgelist",comments="@")
						nx.write_graphml(claim_g,filename+".graphml",prettyprint=True)
						#plot claim graph
						# plotFredGraph(claim_g,filename+".png")
					else:
						errorclaimid.append(filename.split("/")[-1].strip(".rdf"))
					break
				else:
					diff2=min(abs(minsec-dif2),60)
					print("Sleeping for ",round(diff2))
					time.sleep(abs(diff2))
					start2=time.time()
			except xml.sax._exceptions.SAXParseException:
				print("Exception Occurred")
				errorclaimid.append(claim_ID)
				break
	return errorclaimid,clean_claims

#Function to passively parse existing fred rdf files
def passiveFredParse(index,claims_path,claim_IDs,init,end):
	end=min(end,len(claim_IDs))
	#Reading claim_IDs
	errorclaimid=[]
	rdf=rdflib.Graph()
	clean_claims={}
	for i in range(init,end):
		claim_ID=claim_IDs[i]
		filename=os.path.join(claims_path,"claim{}".format(str(claim_ID)))
		try:
			g=openFredGraph(filename+".rdf")
		except:
			print("Exception Occurred")
			errorclaimid.append(claim_ID)
			continue
		claim_g,nodes2remove,nodes2contract=checkClaimGraph(g,'rdf')
		#store pruning data
		clean_claims[str(claim_ID)]={}
		clean_claims[str(claim_ID)]['nodes2remove']=nodes2remove
		clean_claims[str(claim_ID)]['nodes2contract']=nodes2contract
		#write pruning data
		with codecs.open(filename+"_clean.json","w","utf-8") as f:
			f.write(json.dumps(clean_claims[str(claim_ID)],indent=4,ensure_ascii=False))
		#write claim graph as edgelist and graphml
		nx.write_edgelist(claim_g,filename+".edgelist")
		try:
			claim_g=nx.read_edgelist(filename+".edgelist",comments="@")
		except TypeError:
			print(claim_ID)
			break
		nx.write_graphml(claim_g,filename+".graphml",prettyprint=True)
		#plot claim graph
		# plotFredGraph(claim_g,filename+".png")
	return index,errorclaimid,clean_claims

#Function to return a clean graph, depending on the edges to delete and contract
def cleanClaimGraph(claim_g,clean_claims):
	nodes2remove=clean_claims['nodes2remove']
	nodes2contract=clean_claims['nodes2contract']
	for node in nodes2remove['%27']:
		if claim_g.has_node(node):
			claim_g.remove_node(node)
	for nodes in nodes2contract['type']:
		if claim_g.has_node(nodes[0]) and claim_g.has_node(nodes[1]):
			claim_g=nx.contracted_nodes(claim_g,nodes[0],nodes[1],self_loops=False)
	for nodes in nodes2contract['sub']:
		if claim_g.has_node(nodes[0]) and claim_g.has_node(nodes[1]):
			claim_g=nx.contracted_nodes(claim_g,nodes[0],nodes[1],self_loops=False)
	for nodes in nodes2contract['vnequiv']:
		if claim_g.has_node(nodes[0]) and claim_g.has_node(nodes[1]):
			claim_g=nx.contracted_nodes(claim_g,nodes[0],nodes[1],self_loops=False)
	for nodes in nodes2contract['dbpediasas']:
		if claim_g.has_node(nodes[0]) and claim_g.has_node(nodes[1]):
			claim_g=nx.contracted_nodes(claim_g,nodes[0],nodes[1],self_loops=False)
	for nodes in nodes2contract['dbpediaequiv']:
		if claim_g.has_node(nodes[0]) and claim_g.has_node(nodes[1]):
			claim_g=nx.contracted_nodes(claim_g,nodes[0],nodes[1],self_loops=False)
	for nodes in nodes2contract['dbpediassoc']:
		if claim_g.has_node(nodes[0]) and claim_g.has_node(nodes[1]):
			claim_g=nx.contracted_nodes(claim_g,nodes[0],nodes[1],self_loops=False)
	for node in nodes2remove['det']:
		if claim_g.has_node(node):
			claim_g.remove_node(node)
	for node in nodes2remove['data']:
		if claim_g.has_node(node):
			claim_g.remove_node(node)
	for node in nodes2remove['prop']:
		if claim_g.has_node(node):
			claim_g.remove_node(node)
	for node in nodes2remove['url']:
		if claim_g.has_node(node):
			claim_g.remove_node(node)
	claim_g.remove_nodes_from(list(nx.isolates(claim_g)))
	return claim_g

#Function save individual claim graphs
def saveClaimGraph(claim_g,filename,cf,i):
	nx.write_edgelist(claim_g,filename+"_clean{}.edgelist".format(str(cf)+'_'+str(i)))
	claim_g=nx.read_edgelist(filename+"_clean{}.edgelist".format(str(cf)+'_'+str(i)),comments="@")
	nx.write_graphml(claim_g,filename+"_clean{}.graphml".format(str(cf)+'_'+str(i)),prettyprint=True)
	# plotFredGraph(claim_g,filename,cf)

#Function to stitch/compile graphs in an iterative way. i.e clean individual graphs before unioning 
def compileClaimGraph1(index,claims_path,claim_IDs,clean_claims,init,end):
	end=min(end,len(claim_IDs))
	fcg=nx.Graph()
	for claim_ID in claim_IDs[init:end]:
		filename=os.path.join(claims_path,"claim{}".format(str(claim_ID)))
		try:
			claim_g=nx.read_edgelist(filename+".edgelist",comments="@")
		except:
			continue
		claim_g=cleanClaimGraph(claim_g,clean_claims[str(claim_ID)])
		flatten = lambda l: [item for sublist in l for item in sublist]
		while True:
			claim_g,nodes2remove,nodes2contract=checkClaimGraph(claim_g,'nx')
			clean_claims[str(claim_ID)]['nodes2remove']=nodes2remove
			clean_claims[str(claim_ID)]['nodes2contract']=nodes2contract
			if bool(flatten(flatten_dict(clean_claims[str(claim_ID)]).values()))==False:
				break;
			claim_g=cleanClaimGraph(claim_g,clean_claims[str(claim_ID)])
			# import pdb
			# pdb.set_trace()
		# saveClaimGraph(claim_g,filename,1)
		fcg=nx.compose(fcg,claim_g)
	return index,fcg

#Function to stitch/compile graphs in a one shot way. i.e clean entire graph after unioning 
def compileClaimGraph2(index,claims_path,claim_IDs,clean_claims,init,end):
	end=min(end,len(claim_IDs))
	fcg=nx.Graph()
	for claim_ID in claim_IDs[init:end]:
		filename=os.path.join(claims_path,"claim{}".format(str(claim_ID)))
		try:
			claim_g=nx.read_edgelist(filename+".edgelist",comments="@")
		except:
			continue
		fcg=nx.compose(fcg,claim_g)
	return index,fcg

#Function to stitch/compile graphs in a hybrid way. i.e clean entire graph after unioning with each claim graph
def compileClaimGraph3(claims_path,claim_IDs,clean_claims):
	fcg=nx.Graph()
	for claim_ID in claim_IDs:
		filename=os.path.join(claims_path,"claim{}".format(str(claim_ID)))
		try:
			claim_g=nx.read_edgelist(filename+".edgelist",comments="@")
		except:
			continue
		fcg=nx.compose(fcg,claim_g)
		fcg=cleanClaimGraph(fcg,clean_claims[str(claim_ID)])
	return fcg

#Function to aggregate the graph cleaning dictionary for the entire graph
def compile_clean(rdf_path,clean_claims,claim_type):
	master_clean={}
	master_clean['nodes2contract']={}
	master_clean['nodes2remove']={}
	master_clean['nodes2remove']['%27']=[]
	master_clean['nodes2remove']['det']=[]
	master_clean['nodes2remove']['url']=[]
	master_clean['nodes2remove']['prop']=[]
	master_clean['nodes2remove']['data']=[]
	master_clean['nodes2contract']['type']=[]
	master_clean['nodes2contract']['vnequiv']=[]
	master_clean['nodes2contract']['dbpediasas']=[]
	master_clean['nodes2contract']['dbpediaequiv']=[]
	master_clean['nodes2contract']['dbpediassoc']=[]
	with codecs.open(os.path.join(rdf_path,"{}claims_clean.txt".format(claim_type)),"w","utf-8") as f: 
		pprint(clean_claims,stream=f)
	for clean_claim in clean_claims.values():
		for key in clean_claim.keys():
			for key2 in clean_claim[key].keys():
				master_clean[key][key2]+=clean_claim[key][key2]
	with codecs.open(os.path.join(rdf_path,"{}master_clean.txt".format(claim_type)),"w","utf-8") as f: 
		pprint(master_clean,stream=f)
	with codecs.open(os.path.join(rdf_path,"{}master_clean.json".format(claim_type)),"w","utf-8") as f:
		f.write(json.dumps(master_clean,indent=4,ensure_ascii=False))
	return master_clean

#Function to save fred graph including its nodes, entities, node2ID dictionary and edgelistID (format needed by klinker)	
def saveFred(fcg,graph_path,fcg_label,compilefred):
	fcg_path=os.path.join(graph_path,"fred"+str(compilefred),fcg_label)
	os.makedirs(fcg_path, exist_ok=True)
	#writing aggregated networkx graphs as edgelist and graphml
	nx.write_edgelist(fcg,os.path.join(fcg_path,"{}.edgelist".format(fcg_label)))
	fcg=nx.read_edgelist(os.path.join(fcg_path,"{}.edgelist".format(fcg_label)),comments="@")
	#Saving graph as graphml
	nx.write_graphml(fcg,os.path.join(fcg_path,"{}.graphml".format(fcg_label)),prettyprint=True)
	os.makedirs(os.path.join(fcg_path,"data"),exist_ok=True)
	write_path=os.path.join(fcg_path,"data",fcg_label)
	nodes=list(fcg.nodes)
	edges=list(fcg.edges)
	#Save Nodes
	with codecs.open(write_path+"_nodes.txt","w","utf-8") as f:
		for node in nodes:
			f.write(str(node)+"\n")
	#Save Entities
	entity_regex=re.compile(r'http:\/\/dbpedia\.org')
	entities=np.asarray([node for node in nodes if entity_regex.match(node)])
	with codecs.open(write_path+"_entities.txt","w","utf-8") as f:
		for entity in entities:
			f.write(str(entity)+"\n")
	#Save node2ID dictionary
	node2ID={node:i for i,node in enumerate(nodes)}
	with codecs.open(write_path+"_node2ID.json","w","utf-8") as f:
		f.write(json.dumps(node2ID,indent=4,ensure_ascii=False))
	#Save Edgelist ID
	edgelistID=np.asarray([[int(node2ID[edge[0]]),int(node2ID[edge[1]]),1] for edge in edges])
	np.save(write_path+"_edgelistID.npy",edgelistID)  


def createFred(rdf_path,graph_path,fcg_label,init,passive,cpu,compilefred):
	fcg_path=os.path.join(graph_path,"fred"+str(compilefred),fcg_label+str(compilefred))
	#If union of tfcg and ffcg wants to be created i.e ufcg
	if fcg_label=="ufcg":
		#Assumes that tfcg and ffcg exists
		tfcg_path=os.path.join(graph_path,"fred"+str(compilefred),"tfcg"+str(compilefred),"tfcg{}.edgelist".format(str(compilefred)))
		ffcg_path=os.path.join(graph_path,"fred"+str(compilefred),"ffcg"+str(compilefred),"ffcg{}.edgelist".format(str(compilefred)))
		if os.path.exists(tfcg_path) and os.path.exists(ffcg_path):
			tfcg=nx.read_edgelist(tfcg_path,comments="@")
			ffcg=nx.read_edgelist(ffcg_path,comments="@")
			ufcg=nx.compose(tfcg,ffcg)
			os.makedirs(fcg_path, exist_ok=True)
			saveFred(ufcg,graph_path,fcg_label+str(compilefred),compilefred)
		else:
			print("Create tfcg and ffcg before attempting to create the union: ufcg")
	else:
		claim_types={"tfcg":"true","ffcg":"false"}
		claim_type=claim_types[fcg_label]
		claims_path=os.path.join(rdf_path,"{}_claims".format(claim_type))
		claim_IDs=np.load(os.path.join(rdf_path,"{}_claimID.npy".format(claim_type)))
		claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)),index_col=0)
		#compiling fred only i.e. stitching together the graph using the dictionary that stores edges to remove and contract i.e. clean_claims
		if compilefred!=0:
			with codecs.open(os.path.join(rdf_path,"{}claims_clean.json".format(claim_type)),"r","utf-8") as f: 
				clean_claims=json.loads(f.read())
			if compilefred==1 or compilefred==2:
				n=int(len(claim_IDs)/cpu)+1
				pool=mp.Pool(processes=cpu)							
				results=[pool.apply_async(eval("compileClaimGraph"+str(compilefred)), args=(index,claims_path,claim_IDs,clean_claims,index*n,(index+1)*n)) for index in range(cpu)]
				output=sorted([p.get() for p in results],key=lambda x:x[0])
				fcgs=list(map(lambda x:x[1],output))
				master_fcg=nx.Graph()
				for fcg in fcgs:
					master_fcg=nx.compose(master_fcg,fcg)
				if compilefred==2:
					master_clean=compile_clean(rdf_path,clean_claims,claim_type)
					master_fcg=cleanClaimGraph(master_fcg,master_clean)
			elif compilefred==3:
				master_fcg=compileClaimGraph3(claims_path,claim_IDs,clean_claims)
			saveFred(master_fcg,graph_path,fcg_label+str(compilefred),compilefred)
		#else parsing the graph for each claim
		else:
			#passive: if graph rdf files have already been fetched from fred. Faster parallelizable
			if passive:
				n=int(len(claim_IDs)/cpu)+1
				pool=mp.Pool(processes=cpu)							
				results=[pool.apply_async(passiveFredParse, args=(index,claims_path,claim_IDs,index*n,(index+1)*n)) for index in range(cpu)]
				output=sorted([p.get() for p in results],key=lambda x:x[0])
				errorclaimid=list(chain(*map(lambda x:x[1],output)))
				clean_claims=dict(ChainMap(*map(lambda x:x[2],output)))
			#if graph rdf files have not been fetched. Slower, dependent on rate limits
			else:
				errorclaimid,clean_claims=fredParse(claims_path,claims,init,end)
			np.save(os.path.join(rdf_path,"{}_error_claimID.npy".format(fcg_label)),errorclaimid)
			with codecs.open(os.path.join(rdf_path,"{}claims_clean.json".format(claim_type)),"w","utf-8") as f:
				f.write(json.dumps(clean_claims,indent=4,ensure_ascii=False))

#Function to plot a networkx graph
def plotFredGraph(claim_g,filename,cf):
	plt.figure()
	pos=nx.spring_layout(claim_g)
	nx.draw(claim_g,pos,labels={node:node.split("/")[-1].split("#")[-1] for node in claim_g.nodes()},node_size=400)
	edge_labels={(edge[0], edge[1]): edge[2]['label'].split("/")[-1].split("#")[-1] for edge in claim_g.edges(data=True)}
	nx.draw_networkx_edge_labels(claim_g,pos,edge_labels)
	plt.axis('off')
	plt.savefig(filename+"_clean"+str(cf))
	plt.close()
	plt.clf()

# if __name__== "__main__":
# 	parser=argparse.ArgumentParser(description='Create fred graph')
# 	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/rdf_files/')
# 	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
# 	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg','ffcg','ufcg'],help='True False or Union FactCheckGraph')
# 	parser.add_argument('-i','--init', metavar='Index Start',type=int,help='Index number of claims to start from',default=0)
# 	parser.add_argument('-p','--passive',action='store_true',help='Passive or not',default=False)
# 	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
# 	parser.add_argument('-cf','--compilefred',metavar='Compile method #',type=int,help='Number of compile method',default=0)
# 	args=parser.parse_args()
# 	createFred(args.rdfpath,args.graphpath,args.fcgtype,args.init,args.passive,args.cpu,args.compilefred)




