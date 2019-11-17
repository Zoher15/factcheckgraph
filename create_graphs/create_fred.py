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
matplotlib.use('Agg')
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

__author__='Misael Mongiovi, Andrea Giovanni Nuzzolese'
# Original script edited by Zoher Kachwala for FactCheckGraph

plugin.register('application/rdf+xml', Serializer, 'rdflib.plugins.serializers.rdfxml', 'XMLSerializer')
plugin.register('xml', Serializer, 'rdflib.plugins.serializers.rdfxml', 'XMLSerializer')

class FredType(Enum):
	Situation=1
	Event=2
	NamedEntity=3
	SkolemizedEntity=4
	Quality=5
	Concept=6

class NodeType(Enum):
	Class=1
	Instance=0

class ResourceType(Enum):
	Fred=0
	Dbpedia=1
	Verbnet=2

class EdgeMotif(Enum):
	Identity=1
	Type=2
	SubClass=3
	Equivalence=4
	Role=5
	Modality=6
	Negation=7
	Property=8

class NaryMotif(Enum):
	Event=1
	Situation=2
	OtherEvent=3
	Concept=4

class PathMotif(Enum):
	Type=1
	SubClass=2

class ClusterMotif(Enum):
	Identity=1
	Equivalence=2
	IdentityEquivalence=3 #all concepts tied by a sequence of sameAs and equivalentClass in any direction

class Role(Enum):
	Agent=1
	Patient=2
	Theme=3
	Location=4
	Time=5
	Involve=6
	Declared=7
	VNOblique=8
	LocOblique=9
	ConjOblique=10
	Extended=11
	Associated=12

class FredNode(object):
	def __init__(self,nodetype,fredtype,resourcetype):
		self.Type=nodetype
		self.FredType=fredtype
		self.ResourceType=resourcetype

class FredEdge(object):
	def __init__(self,edgetype):
		self.Type=edgetype


class FredGraph:
	def __init__(self,rdf):
		self.rdf=rdf

	def getNodes(self):
		nodes=set()
		for a, b, c in self.rdf:
			nodes.add(a.strip())
			nodes.add(c.strip())
		return nodes

	def getClassNodes(self):
		query="PREFIX owl: <http://www.w3.org/2002/07/owl#> " \
				"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
				"SELECT ?t WHERE { " \
				"?i a ?t1 . " \
				"?t1 (owl:equivalentClass | ^owl:equivalentClass | rdfs:sameAs | ^rdfs:sameAs | rdfs:subClassOf)* ?t }"

		nodes=set()
		res=self.rdf.query(query)
		for el in res:
			nodes.add(el[0].strip())
		return nodes

	def getInstanceNodes(self):
		nodes=self.getNodes()
		return nodes.difference(self.getClassNodes())

	def getEventNodes(self):
		query="PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#> " \
				"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
				"PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
				"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
				"SELECT ?e WHERE { ?e a ?t . ?t rdfs:subClassOf* dul:Event }"

		nodes=set()
		res=self.rdf.query(query)
		for el in res:
			nodes.add(el[0].strip())
		return nodes

	def getSituationNodes(self):
		query="PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#> " \
				"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
				"PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
				"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
				"SELECT ?e WHERE { ?e a ?t . ?t rdfs:subClassOf* boxing:Situation }"

		nodes=set()
		res=self.rdf.query(query)
		for el in res:
			nodes.add(el[0].strip())
		return nodes

	def getNamedEntityNodes(self):
		nodes=self.getNodes()
		events=self.getEventNodes()
		classes=self.getClassNodes()
		qualities=self.getQualityNodes()
		situation=self.getSituationNodes()

		ne=set()
		for n in nodes:
			if n not in classes and n not in qualities and n not in events and n not in situation:
				suffix=n[n.find("_", -1):]
				if suffix.isdigit()==False:
					ne.add(n)
		return ne

	def getSkolemizedEntityNodes(self):
		nodes=self.getNodes()
		events=self.getEventNodes()
		classes=self.getClassNodes()
		qualities=self.getQualityNodes()
		situation=self.getSituationNodes()

		ne=set()
		for n in nodes:
			if n not in classes and n not in qualities and n not in events and n not in situation:
				suffix=n[n.find("_", -1):]
				if suffix.isdigit()==True:
					ne.add(n)
		return ne

	def getQualityNodes(self):
		query="PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
				"SELECT ?q WHERE { ?i dul:hasQuality ?q }"
		nodes=set()
		res=self.rdf.query(query)
		for el in res:
			nodes.add(el[0].strip())
		return nodes

	def getConceptNodes(self):
		return self.getClassNodes()

	#return 1 for situation, 2 for event, 3 for named entity, 4 for concept class, 5 for concept instance
	def getInfoNodes(self):

		def getResource(n):
			if node.find("http://www.ontologydesignpatterns.org/ont/dbpedia/")==0:
				return ResourceType.Dbpedia
			elif node.find("http://www.ontologydesignpatterns.org/ont/vn/")==0:
				return ResourceType.Verbnet
			else:
				return ResourceType.Fred

		nodes=dict()
		query="PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#> " \
				"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
				"PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
				"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
				"PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
				"SELECT ?n ?class ?x WHERE { { ?n a ?t . ?t rdfs:subClassOf* boxing:Situation bind (1 as ?x) bind (0 as ?class) } " \
				"UNION {?n a ?t . ?t rdfs:subClassOf* dul:Event bind (2 as ?x)  bind (0 as ?class)} " \
				"UNION {?i a ?t . ?t (owl:equivalentClass | ^owl:equivalentClass | rdfs:sameAs | ^rdfs:sameAs | rdfs:subClassOf)* ?n bind (6 as ?x) bind (1 as ?class)} }"

		res=self.rdf.query(query)
		for el in res:
			node=el[0].strip()
			cl=NodeType(el[1].value)
			type=FredType(el[2].value)
			nodes[node]=FredNode(cl,type,getResource(node))

		#if not an event nor situation nor class

		qualities=self.getQualityNodes()
		for n in qualities:
			if n not in nodes:
				nodes[n]=FredNode(NodeType.Instance,FredType.Quality,getResource(n))

		#if not even quality

		for n in self.getNodes():
			if n not in nodes:
				suffix=n[n.find("_", -1):]
				if n not in qualities and suffix.isdigit()==False:
					nodes[n]=FredNode(NodeType.Instance,FredType.NamedEntity,getResource(n))
				else:
					nodes[n]=FredNode(NodeType.Instance,FredType.SkolemizedEntity,getResource(n))

		return nodes

	def getEdges(self):
		return [(a.strip(),b.strip(),c.strip()) for (a,b,c) in self.rdf]

	#def getRoleEdges(self):
	#	return self.getEdgeMotif(EdgeMotif.Role)

	def getEdgeMotif(self,motif):
		if motif==EdgeMotif.Role:
			query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"SELECT ?i ?p ?o ?r WHERE " \
					"{?i ?p ?o . ?i a ?t . ?t rdfs:subClassOf* dul:Event BIND (5 as ?r) }"
		elif motif==EdgeMotif.Identity:
			query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
					"SELECT ?i ?p ?o ?r WHERE " \
					"{?i ?p ?o . FILTER(?p=owl:sameAs ) BIND (1 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}}"
		elif motif==EdgeMotif.Type:
			query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"SELECT ?i ?p ?o ?r WHERE " \
					"{?i ?p ?o . FILTER(?p=rdf:type ) BIND (2 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}}"
		elif motif==EdgeMotif.SubClass:
			query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"SELECT ?i ?p ?o ?r WHERE " \
					"{?i ?p ?o . FILTER(?p=rdfs:subClassOf ) BIND (3 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}}"
		elif motif==EdgeMotif.Equivalence:
			query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
					"SELECT ?i ?p ?o ?r WHERE " \
					"{?i ?p ?o . FILTER(?p=owl:equivalentClass ) BIND (4 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}}"
		elif motif==EdgeMotif.Modality:
			query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
					"SELECT ?i ?p ?o ?r WHERE " \
					"{?i ?p ?o . FILTER(?p=boxing:hasModality ) BIND (6 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}}"
		elif motif==EdgeMotif.Negation:
			query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
					"SELECT ?i ?p ?o ?r WHERE " \
					"{?i ?p ?o . FILTER(?p=boxing:hasTruthValue ) BIND (7 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}}"
		elif motif==EdgeMotif.Property:
			query="PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
					"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
					"SELECT ?i ?p ?o ?r WHERE " \
					"{?i ?p ?o . " \
					"FILTER((?p != owl:sameAs) && (?p != rdf:type) && (?p != rdfs:subClassOf) && (?p != owl:equivalentClass) && (?p != boxing:hasModality) && (?p != boxing:hasTruthValue)) " \
					"FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event} " \
					"BIND (8 as ?r) }"
		else:
			raise Exception("Unknown motif: " + str(motif))

		return [(el[0].strip(),el[1].strip(),el[2].strip()) for el in self.rdf.query(query)]

	def getPathMotif(self,motif):
		if motif==PathMotif.Type:
			query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"SELECT ?i ?o WHERE " \
					"{?i rdf:type ?t . ?t rdfs:subClassOf* ?o}"
		elif motif==PathMotif.SubClass:
			query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"SELECT ?i ?o WHERE " \
					"{?i rdfs:subClassOf+ ?o}"
		else:
			raise Exception("Unknown motif: " + str(motif))

		return [(el[0].strip(),el[1].strip()) for el in self.rdf.query(query)]

	def getClusterMotif(self,motif):
		if motif==ClusterMotif.Identity:
			query="PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
					"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"SELECT DISTINCT ?s ?o WHERE " \
					"{ ?s (owl:sameAs|^owl:sameAs)+ ?o } ORDER BY ?s "
		elif motif==ClusterMotif.Equivalence:
			query="PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
					"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"SELECT DISTINCT ?s ?o WHERE " \
					"{ ?s (^owl:equivalentClass|owl:equivalentClass)+ ?o } ORDER BY ?s "
		elif motif==ClusterMotif.IdentityEquivalence:
			query="PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
					"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
					"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> " \
					"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
					"SELECT DISTINCT ?s ?o WHERE " \
					"{ ?s (owl:sameAs|^owl:sameAs|^owl:equivalentClass|owl:equivalentClass)+ ?o } ORDER BY ?s "
		else:
			raise Exception("Unknown motif: " + str(motif))

		results=self.rdf.query(query)

		clusters=list()
		used=set()
		olds=None
		currentset=set()
		for el in results:
			s=el[0].strip()
			o=el[1].strip()
			if s != olds:
				if len(currentset) != 0:
					currentset.add(olds)
					clusters.append(currentset)
					used=used.union(currentset)
					currentset=set()
				fillSet=False if s in used else True
			if fillSet==True:
				currentset.add(o)
			olds=s

		if len(currentset) != 0:
			currentset.add(olds)
			clusters.append(currentset)

		return clusters

	def getInfoEdges(self):
		edges=dict()
		query="PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#> " \
				"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
				"PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
				"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
				"PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
				"" \
				"SELECT ?i ?p ?o ?r WHERE {" \
				"{?i ?p ?o . ?i a ?t . ?t rdfs:subClassOf* dul:Event BIND (5 as ?r) }" \
				"UNION" \
				"{?i ?p ?o . FILTER(?p=owl:sameAs ) BIND (1 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}  }" \
				"UNION" \
				"{?i ?p ?o . FILTER(?p=rdf:type ) BIND (2 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}  }" \
				"UNION" \
				"{?i ?p ?o . FILTER(?p=rdfs:subClassOf ) BIND (3 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}  }" \
				"UNION" \
				"{?i ?p ?o . FILTER(?p=owl:equivalentClass ) BIND (4 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}  }" \
				"UNION" \
				"{?i ?p ?o . FILTER(?p=boxing:hasModality ) BIND (6 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}  }" \
				"UNION" \
				"{?i ?p ?o . FILTER(?p=boxing:hasTruthValue ) BIND (7 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}  }" \
				"}"

		res=self.rdf.query(query)
		for el in res:
			edges[(el[0].strip(),el[1].strip(),el[2].strip())]=FredEdge(EdgeMotif(el[3].value))
		for e in self.getEdges():
			if e not in edges:
				edges[e]=FredEdge(EdgeMotif.Property)
		return edges

	def getNaryMotif(self,motif):
		def fillRoles(el):
			relations=dict()
			if el['agent'] != None:
				relations[Role.Agent]=el['agent']
			if el['patient'] != None:
				relations[Role.Patient]=el['patient']
			if el['theme'] != None:
				relations[Role.Theme]=el['theme']
			if el['location'] != None:
				relations[Role.Theme]=el['location']
			if el['time'] != None:
				relations[Role.Theme]=el['time']
			if el['involve'] != None:
				relations[Role.Theme]=el['involve']
			if el['declared'] != None:
				relations[Role.Theme]=el['declared']
			if el['vnoblique'] != None:
				relations[Role.Theme]=el['vnoblique']
			if el['locoblique'] != None:
				relations[Role.Theme]=el['locoblique']
			if el['conjoblique'] != None:
				relations[Role.Theme]=el['conjoblique']
			if el['extended'] != None:
				relations[Role.Theme]=el['extended']
			if el['associated'] != None:
				relations[Role.Theme]=el['associated']
			return relations

		query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
				"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> " \
				"PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>" \
				"PREFIX vnrole: <http://www.ontologydesignpatterns.org/ont/vn/abox/role/>" \
				"PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#>" \
				"PREFIX boxer: <http://www.ontologydesignpatterns.org/ont/boxer/boxer.owl#>" \
				"PREFIX d0: <http://www.ontologydesignpatterns.org/ont/d0.owl#>" \
				"PREFIX schemaorg: <http://schema.org/>" \
				"PREFIX earmark: <http://www.essepuntato.it/2008/12/earmark#>" \
				"PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#>" \
				"PREFIX wn: <http://www.w3.org/2006/03/wn/wn30/schema/>" \
				"PREFIX pos: <http://www.ontologydesignpatterns.org/ont/fred/pos.owl#>" \
				"PREFIX semiotics: <http://ontologydesignpatterns.org/cp/owl/semiotics.owl#>" \
				"PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
				"SELECT DISTINCT" \
				"?node ?type " \
				"?agentiverole ?agent" \
				"?passiverole ?patient" \
				"?themerole ?theme" \
				"?locativerole ?location" \
				"?temporalrole ?time" \
				"?situationrole ?involve" \
				"?declarationrole ?declared" \
				"?vnobrole ?vnoblique" \
				"?preposition ?locoblique" \
				"?conjunctive ?conjoblique" \
				"?periphrastic ?extended" \
				"?associationrole ?associated " \
				"WHERE " \
				"{" \
				"{{?node rdf:type ?concepttype bind (4 as ?type)" \
				"MINUS {?node rdf:type rdf:Property}" \
				"MINUS {?node rdf:type owl:ObjectProperty}" \
				"MINUS {?node rdf:type owl:DatatypeProperty}" \
				"MINUS {?node rdf:type owl:Class}" \
				"MINUS {?node rdf:type earmark:PointerRange}" \
				"MINUS {?node rdf:type earmark:StringDocuverse}" \
				"MINUS {?concepttype rdfs:subClassOf+ dul:Event}" \
				"MINUS {?node rdf:type boxing:Situation}" \
				"MINUS {?concepttype rdfs:subClassOf+ schemaorg:Event}" \
				"MINUS {?concepttype rdfs:subClassOf+ d0:Event}}" \
				"}" \
				"UNION " \
				" {?node rdf:type boxing:Situation bind (2 as ?type)}" \
				"UNION" \
				" {?node rdf:type ?verbtype . ?verbtype rdfs:subClassOf* dul:Event bind (1 as ?type)}" \
				"UNION" \
				" {?node rdf:type ?othereventtype . ?othereventtype rdfs:subClassOf* schemaorg:Event bind (3 as ?type)}" \
				"UNION" \
				" {?node rdf:type ?othereventtype . ?othereventtype rdfs:subClassOf* d0:Event bind (3 as ?type)}" \
				"OPTIONAL " \
				" {?node ?agentiverole ?agent" \
				" FILTER (?agentiverole=vnrole:Agent || ?agentiverole=vnrole:Actor1 || ?agentiverole=vnrole:Actor2 || ?agentiverole=vnrole:Experiencer || ?agentiverole=vnrole:Cause || ?agentiverole=boxer:agent)}" \
				"OPTIONAL " \
				" {?node ?passiverole ?patient" \
				" FILTER (?passiverole=vnrole:Patient || ?passiverole=vnrole:Patient1 || ?passiverole=vnrole:Patient2 || ?passiverole=vnrole:Beneficiary || ?passiverole=boxer:patient || ?passiverole=vnrole:Recipient || ?passiverole=boxer:recipient)} " \
				"OPTIONAL " \
				" {?node ?themerole ?theme" \
				" FILTER (?themerole=vnrole:Theme || ?themerole=vnrole:Theme1 || ?themerole=vnrole:Theme2 || ?themerole=boxer:theme)} " \
				"OPTIONAL " \
				" {?node ?locativerole ?location" \
				" FILTER (?locativerole=vnrole:Location || ?locativerole=vnrole:Destination || ?locativerole=vnrole:Source || ?locativerole=fred:locatedIn)} " \
				"OPTIONAL " \
				" {?node ?temporalrole ?time" \
				" FILTER (?temporalrole=vnrole:Time)} " \
				"OPTIONAL " \
				" {?node ?situationrole ?involve" \
				" FILTER (?situationrole=boxing:involves)} " \
				"OPTIONAL " \
				" {?node ?declarationrole ?declared" \
				" FILTER (?declarationrole=boxing:declaration || ?declarationrole=vnrole:Predicate || ?declarationrole=vnrole:Proposition)} " \
				"OPTIONAL " \
				" { ?node ?vnobrole ?vnoblique " \
				" FILTER (?vnobrole=vnrole:Asset || ?vnobrole=vnrole:Attribute || ?vnobrole=vnrole:Extent || ?vnobrole=vnrole:Instrument || ?vnobrole=vnrole:Material || ?vnobrole=vnrole:Oblique || ?vnobrole=vnrole:Oblique1 || ?vnobrole=vnrole:Oblique2 || ?vnobrole=vnrole:Product || ?vnobrole=vnrole:Stimulus || ?vnobrole=vnrole:Topic || ?vnobrole=vnrole:Value)}" \
				"OPTIONAL " \
				" {?node ?preposition ?locoblique" \
				" FILTER (?preposition=fred:about || ?preposition=fred:after || ?preposition=fred:against || ?preposition=fred:among || ?preposition=fred:at || ?preposition=fred:before || ?preposition=fred:between || ?preposition=fred:by || ?preposition=fred:concerning || ?preposition=fred:for || ?preposition=fred:from || ?preposition=fred:in || ?preposition=fred:in_between || ?preposition=fred:into || ?preposition=fred:of || ?preposition=fred:off || ?preposition=fred:on || ?preposition=fred:onto || ?preposition=fred:out_of || ?preposition=fred:over || ?preposition=fred:regarding || ?preposition=fred:respecting || ?preposition=fred:through || ?preposition=fred:to || ?preposition=fred:towards || ?preposition=fred:under || ?preposition=fred:until || ?preposition=fred:upon || ?preposition=fred:with)}" \
				"OPTIONAL " \
				" {{?node ?conjunctive ?conjoblique" \
				" FILTER (?conjunctive=fred:as || ?conjunctive=fred:when || ?conjunctive=fred:after || ?conjunctive=fred:where || ?conjunctive=fred:whenever || ?conjunctive=fred:wherever || ?conjunctive=fred:because || ?conjunctive=fred:if || ?conjunctive=fred:before || ?conjunctive=fred:since || ?conjunctive=fred:unless || ?conjunctive=fred:until || ?conjunctive=fred:while)} UNION {?conjoblique ?conjunctive ?node FILTER (?conjunctive=fred:once || ?conjunctive=fred:though || ?conjunctive=fred:although)}}" \
				"OPTIONAL " \
				" {?node ?periphrastic ?extended" \
				" FILTER (?periphrastic != ?vnobrole && ?periphrastic != ?preposition && ?periphrastic != ?conjunctive && ?periphrastic != ?agentiverole && ?periphrastic != ?passiverole && ?periphrastic != ?themerole && ?periphrastic != ?locativerole && ?periphrastic != ?temporalrole && ?periphrastic != ?situationrole && ?periphrastic != ?declarationrole && ?periphrastic != ?associationrole && ?periphrastic != boxing:hasTruthValue && ?periphrastic != boxing:hasModality && ?periphrastic != dul:hasQuality && ?periphrastic != dul:associatedWith && ?periphrastic != dul:hasRole &&?periphrastic != rdf:type)}" \
				"OPTIONAL " \
				" {?node ?associationrole ?associated" \
				" FILTER (?associationrole=boxer:rel || ?associationrole=dul:associatedWith)} " \
				"}" \
				" ORDER BY ?type"

		results=self.rdf.query(query)
		motifocc=dict()
		for el in results:
			if NaryMotif(el['type'])==motif:
				motifocc[el['node'].strip()]=fillRoles(el)

		return motifocc

	def getCompactGraph(self):
		pass

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

def checkFredFile(filename):
	g=openFredGraph(filename)
	checkFredGraph(g)

#function to check a claim and create dictionary of edges to contract and remove
def checkClaimGraph(g):
	regex_assoc=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/dul\/DUL\.owl#associatedWith$')
	regex_freddata_low=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#([a-z]*)_.*')
	regex_freddata_upp=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#([a-zA-Z]*)$')
	regex_27=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#%27.*')
	regex_dul=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/dul\/DUL\.owl#(.*)')
	regex_quant=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/quantifiers\.owl#.*')
	regex_vndata=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/vn\/data\/(.*)')
	regex_dbpedia=re.compile(r'^http:\/\/dbpedia\.org\/resource\/(.*)')
	regex_owl=re.compile(r'^http:\/\/www\.w3\.org\/2002\/07\/owl#(.*)')
	regex_schema=re.compile(r'^http:\/\/schema\.org\/(.*)')
	claim_g=nx.Graph()
	edges2remove={}
	edges2remove['assoc']={}
	edges2remove['assoc']['1']=[]
	edges2remove['assoc']['2']=[]
	edges2remove['assoc']['3']=[]
	edges2remove['assoc']['4']=[]
	edges2remove['misc']=[]
	edges2remove['type']=[]
	edges2contract={}
	edges2contract['ideq']={}
	edges2contract['ideq']['1']=[]
	edges2contract['ideq']['2']=[]
	edges2contract['type']={}
	edges2contract['type']['1']=[]
	edges2contract['type']['2']=[]
	edges2contract['type']['3']=[]
	edges2contract['type']['4']=[]
	for (a,b,c) in g.getEdges():
		claim_g.add_edge(a,c,label=b)
		a_urlparse=urlparse(a)
		c_urlparse=urlparse(c)
		if regex_assoc.match(b):
			# To removed triples like ['http://www.ontologydesignpatterns.org/ont/fred/domain.owl#drop_1', 'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#associatedWith','http://www.ontologydesignpatterns.org/ont/fred/domain.owl#drop_2']
			if regex_freddata_low.match(a) and regex_freddata_low.match(c):
				if regex_freddata_low.match(a)[1]==regex_freddata_low.match(c)[1]:
					edges2remove['assoc']['1'].append((a,b,c))
			elif regex_freddata_low.match(a) and regex_freddata_upp.match(c):
				if regex_freddata_low.match(a)[1]==regex_freddata_upp.match(c)[1].lower():
					edges2remove['assoc']['2'].append((a,b,c))
			elif regex_freddata_low.match(c) and regex_freddata_upp.match(a):
				if regex_freddata_low.match(c)[1]==regex_freddata_upp.match(a)[1].lower():
					edges2remove['assoc']['3'].append((a,b,c))
			else:
				edges2remove['assoc']['4'].append((a,b,c))
		elif regex_quant.match(b) or regex_27.match(a) or regex_27.match(c) or regex_dul.match(a) or regex_dul.match(c) or a==c or (a_urlparse.netloc=='' and a_urlparse.scheme=='') or (c_urlparse.netloc=='' and c_urlparse.scheme==''):
			edges2remove['misc'].append((a,b,c))
	#Merges edges like ['http://www.ontologydesignpatterns.org/ont/fred/domain.owl#million_1','http://www.w3.org/1999/02/22-rdf-syntax-ns#type','http://www.ontologydesignpatterns.org/ont/fred/domain.owl#Million']
	for (a,b) in g.getPathMotif(PathMotif.Type):
		if regex_freddata_low.match(a)!=None and regex_freddata_upp.match(b)!=None:
			if regex_freddata_low.match(a)[1] in regex_freddata_upp.match(b)[1].lower():
				edges2contract['type']['1'].append((b,a))
		elif regex_freddata_low.match(b)!=None and regex_freddata_upp.match(a)!=None:
			if regex_freddata_low.match(b)[1] in regex_freddata_upp.match(a)[1].lower():
				edges2contract['type']['2'].append((a,b))
		elif regex_freddata_low.match(a)!=None and regex_owl.match(b)!=None:
			if regex_freddata_low.match(a)[1] in regex_owl.match(b)[1].lower():
				edges2contract['type']['3'].append((b,a))
		elif regex_freddata_low.match(b)!=None and regex_owl.match(a)!=None:
			if regex_freddata_low.match(b)[1] in regex_owl.match(a)[1].lower():
				edges2contract['type']['4'].append((a,b))
		elif regex_schema.match(a) or regex_schema.match(b):
			edges2remove['type'].append((a,b))

	for cluster in g.getClusterMotif(ClusterMotif.IdentityEquivalence):
		a=list(cluster)[0]
		b=list(cluster)[1]
		if regex_vndata.match(a) or regex_dbpedia.match(a):
			edges2contract['ideq']['1'].append((a,b))
		elif regex_vndata.match(b) or regex_dbpedia.match(b):
			edges2contract['ideq']['1'].append((b,a))
		else:
			edges2contract['ideq']['2'].append((a,b))
	return claim_g,edges2remove,edges2contract

#fetch fred graph files from their API. slow and dependent on rate
def fredParse(claims_path,claims,init,end):
	key="Bearer 56a28f54-7918-3fdd-9d6f-850f13bd4041"
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
						claim_g,edges2remove,edges2contract=checkClaimGraph(g)
						#store pruning data
						clean_claims[str(claim_ID)]={}
						clean_claims[str(claim_ID)]['edges2remove']=edges2remove
						clean_claims[str(claim_ID)]['edges2contract']=edges2contract
						#write pruning data
						with codecs.open(filename+"_clean.json","w","utf-8") as f:
							f.write(json.dumps(clean_claims[str(claim_ID)],ensure_ascii=False))
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
		claim_g,edges2remove,edges2contract=checkClaimGraph(g)
		#store pruning data
		clean_claims[str(claim_ID)]={}
		clean_claims[str(claim_ID)]['edges2remove']=edges2remove
		clean_claims[str(claim_ID)]['edges2contract']=edges2contract
		#write pruning data
		with codecs.open(filename+"_clean.json","w","utf-8") as f:
			f.write(json.dumps(clean_claims[str(claim_ID)],ensure_ascii=False))
		#write claim graph as edgelist and graphml
		nx.write_edgelist(claim_g,filename+".edgelist")
		claim_g=nx.read_edgelist(filename+".edgelist",comments="@")
		nx.write_graphml(claim_g,filename+".graphml",prettyprint=True)
		#plot claim graph
		# plotFredGraph(claim_g,filename+".png")
	return index,errorclaimid,clean_claims

#Function to return a clean graph, depending on the edges to delete and contract
def cleanClaimGraph(claim_g,clean_claims):
	edges2remove=[]
	for key in clean_claims['edges2remove'].keys():
		if type(clean_claims['edges2remove'][key])==dict:
			for key2 in clean_claims['edges2remove'][key].keys():
				edges2remove+=clean_claims['edges2remove'][key][key2]
		else:
			edges2remove+=clean_claims['edges2remove'][key]
	edges2remove=list(map(lambda x:tuple([x[0],x[2]]) if len(x)==3 else tuple(x),edges2remove))
	claim_g.remove_edges_from(edges2remove)
	for key in clean_claims['edges2contract'].keys():
		for key2 in clean_claims['edges2contract'][key].keys():
			if key=='ideq' and key2=='2':
				pass
			else:
				for edge in clean_claims['edges2contract'][key][key2]:
					if claim_g.has_edge(edge[0],edge[1]):
						claim_g=nx.contracted_edge(claim_g,(edge[0],edge[1]),self_loops=False)
	claim_g.remove_nodes_from(list(nx.isolates(claim_g)))
	return claim_g

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
	master_clean['edges2remove']={}
	master_clean['edges2remove']['assoc']={}
	master_clean['edges2remove']['assoc']['1']=[]
	master_clean['edges2remove']['assoc']['2']=[]
	master_clean['edges2remove']['assoc']['3']=[]
	master_clean['edges2remove']['assoc']['4']=[]
	master_clean['edges2remove']['ideq']={}
	master_clean['edges2remove']['ideq']['1']=[]
	master_clean['edges2remove']['ideq']['2']=[]
	master_clean['edges2remove']['misc']=[]
	master_clean['edges2remove']['type']=[]
	master_clean['edges2contract']={}
	master_clean['edges2contract']['ideq']={}
	master_clean['edges2contract']['ideq']['1']=[]
	master_clean['edges2contract']['ideq']['2']=[]
	master_clean['edges2contract']['type']={}
	master_clean['edges2contract']['type']['1']=[]
	master_clean['edges2contract']['type']['2']=[]
	master_clean['edges2contract']['type']['3']=[]
	master_clean['edges2contract']['type']['4']=[]
	with codecs.open(os.path.join(rdf_path,"{}claims_clean.txt".format(claim_type)),"w","utf-8") as f: 
		pprint(clean_claims,stream=f)
	for clean_claim in clean_claims.values():
		for key in clean_claim.keys():
			for key2 in clean_claim[key].keys():
				if type(clean_claim[key][key2])==dict:
					for key3 in clean_claim[key][key2].keys():
						master_clean[key][key2][key3]+=clean_claim[key][key2][key3]
				else:
					master_clean[key][key2]+=clean_claim[key][key2]
	with codecs.open(os.path.join(rdf_path,"{}master_clean.txt".format(claim_type)),"w","utf-8") as f: 
		pprint(master_clean,stream=f)
	with codecs.open(os.path.join(rdf_path,"{}master_clean.json".format(claim_type)),"w","utf-8") as f:
		f.write(json.dumps(master_clean,ensure_ascii=False))
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
		f.write(json.dumps(node2ID,ensure_ascii=False))
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
				f.write(json.dumps(clean_claims,ensure_ascii=False))

#Function to plot a networkx graph
def plotFredGraph(claim_g,filename):
	plt.figure()
	pos=nx.spring_layout(claim_g)
	nx.draw(claim_g,pos,labels={node:node.split("/")[-1].split("#")[-1] for node in claim_g.nodes()},node_size=400)
	edge_labels={(edge[0], edge[1]): edge[2]['label'].split("/")[-1].split("#")[-1] for edge in claim_g.edges(data=True)}
	nx.draw_networkx_edge_labels(claim_g,pos,edge_labels)
	plt.axis('off')
	plt.savefig(filename)
	plt.close()
	plt.clf()

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create fred graph')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/rdf_files/')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg','ffcg','ufcg'],help='True False or Union FactCheckGraph')
	parser.add_argument('-i','--init', metavar='Index Start',type=int,help='Index number of claims to start from',default=0)
	parser.add_argument('-p','--passive',action='store_true',help='Passive or not',default=False)
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	parser.add_argument('-cf','--compilefred',metavar='Compile method #',type=int,help='Number of compile method',default=0)
	args=parser.parse_args()
	createFred(args.rdfpath,args.graphpath,args.fcgtype,args.init,args.passive,args.cpu,args.compilefred)




