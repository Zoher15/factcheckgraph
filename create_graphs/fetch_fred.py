import os
import re
import sys
import time
import rdflib
import requests
import argparse
import random
import networkx as nx
from flufl.enum import Enum
from rdflib import plugin
import pandas as pd
import codecs
from rdflib.serializer import Serializer
from rdflib.plugins.memory import IOMemory
import json
import numpy as np
import xml.sax
import html
from operator import itemgetter
from collections import ChainMap 
from itertools import chain
import multiprocessing as mp
from urllib.parse import urlparse
from pprint import pprint
import re

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
			cl=NodeType[el[1].value]
			type=FredType[el[2].value]
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
			edges[(el[0].strip(),el[1].strip(),el[2].strip())]=FredEdge(EdgeMotif[el[3].value])
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
			if NaryMotif[el['type']]==motif:
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

def nodelabel_mapper(text):
	regex_vn=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/vn\/data\/([a-zA-Z]*)_.*')
	regex_dbpedia=re.compile(r'^http:\/\/dbpedia\.org\/resource\/(.*)')
	regex_quant=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/quantifiers\.owl#.*')
	regex_schema=re.compile(r'^http:\/\/schema\.org.*')
	regex_fred=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#([a-zA-Z]*)_.*')
	regex_fredup=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#([A-Z]+[a-zA-Z]*)')
	try:
		d={
		bool(regex_vn.match(text)):'vn:'+text.split("/")[-1].split("#")[-1],
		bool(regex_dbpedia.match(text)):'db:'+text.split("/")[-1].split("#")[-1],
		bool(regex_quant.match(text)):'quant:'+text.split("/")[-1].split("#")[-1],
		bool(regex_schema.match(text)):'schema:'+text.split("/")[-1].split("#")[-1],
		bool(regex_fred.match(text)):'fred:'+text.split("/")[-1].split("#")[-1],
		bool(regex_fredup.match(text)):'fu:'+text.split("/")[-1].split("#")[-1],
		}
		return d[True]
	except KeyError:
		return 'un:'+text.split("/")[-1].split("#")[-1]

#function to check a claim and create dictionary of nodes to contract and remove
def checkClaimGraph(g,claim_type,claim_ID,graph_type):#,mode):
	claim_g=nx.Graph()
	regex_27=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#%27.*')
	regex_det=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/quantifiers\.owl.*$')
	regex_data=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/dul\/DUL\.owl#hasDataValue$')
	regex_quality=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/dul\/DUL\.owl#hasQuality$')
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
	regex_thing=re.compile(r'^http:\/\/www\.w3\.org\/2002\/07\/owl#Thing$')
	regex_dul=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/dul\/DUL\.owl#(.*)')
	regex_quant=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/quantifiers\.owl#.*')
	regex_schema=re.compile(r'^http:\/\/schema\.org.*')
	nodes2contract={}
	nodes2remove={}
	#########################################################################################
	nodes2contract['type']=set()#keep right node if left node has "_1"
	nodes2contract['subclass']=set()#keep left node
	nodes2contract['equivalence']=set()#keep right node
	nodes2contract['identity']=set()#keep right node
	# nodes2contract['quality']=set()#keep left node
	nodes2remove['num']=set()
	nodes2remove['%27']=set()
	nodes2remove['thing']=set()
	nodes2remove['dul']=set()
	nodes2remove['det']=set()
	nodes2remove['data']=set()
	nodes2remove['prop']=set()
	nodes2remove['schema']=set()#remove schema.org nodes
	edge_list=g.getInfoEdges()
	nodes_set=set()
	ratings={'true':1,'false':-1}
	for e in edge_list:
		(a,b,c)=e
		nodes_set.add(a)
		nodes_set.add(c)
		claim_g.add_edge(a,c,label=b.split("/")[-1].split("#")[-1],claim_ID=claim_ID,rating=ratings[claim_type])
		#Edges
		# if edge_list[e].Type==EdgeMotif.Property:
		# 	if regex_quality.match(b) and (regex_fredup.match(c) and regex_fredup.match(a)):
		# 		if regex_fredup.match(c)[1].lower() in regex_fredup.match(a)[1].lower():
		# 			nodes2contract['quality'].add((a,c))
		if edge_list[e].Type==EdgeMotif.Type or regex_type.match(b):
			if regex_fred.match(a) and (regex_fred.match(a)[1].lower() in c.split("\\")[-1].lower()):
				nodes2contract['type'].add((c,a))
			elif (regex_fred.match(c)) and (regex_fred.match(c)[1].lower() in a.split("\\")[-1].lower()):
				nodes2contract['type'].add((a,c))
		elif edge_list[e].Type==EdgeMotif.SubClass:
			if not regex_dul.match(c):
				if (regex_dbpedia.match(a) and not regex_dbpedia.match(c)) or (regex_vn.match(a) and not regex_vn.match(c)):
					nodes2contract['subclass'].add((a,c))
				else:
					nodes2contract['subclass'].add((c,a))
		elif edge_list[e].Type==EdgeMotif.Equivalence:
			if (regex_dbpedia.match(a) and not regex_dbpedia.match(c)) or (regex_vn.match(a) and not regex_vn.match(c)):
				nodes2contract['equivalence'].add((a,c))
			else:
				nodes2contract['equivalence'].add((c,a))
		elif edge_list[e].Type==EdgeMotif.Identity:
			if (regex_dbpedia.match(a) and not regex_dbpedia.match(c)) or (regex_vn.match(a) and not regex_vn.match(c)):
				nodes2contract['identity'].add((a,c))
			else:
				nodes2contract['identity'].add((c,a))
	for a in nodes_set:
		#Nodes
		a_urlparse=urlparse(a)
		#nodes matching
		if regex_quant.match(a):
			nodes2remove['det'].add(a)
		elif (a_urlparse.netloc=='' and a_urlparse.scheme==''):
			nodes2remove['det'].add(a)
		elif regex_data.match(a):
			nodes2remove['data'].add(a)
		elif regex_prop.match(a):
			nodes2remove['prop'].add(a)
		#Delete '%27' nodes
		elif regex_27.match(a):
			nodes2remove['%27'].add(a)
		elif regex_schema.match(a):
			nodes2remove['schema'].add(a)
		elif regex_dul.match(a):
			nodes2remove['dul'].add(a)
		elif regex_thing.match(a):
			nodes2remove['thing'].add(a)
	nodes2remove={k:sorted(nodes2remove[k]) for k in nodes2remove.keys() if len(nodes2remove[k])>0}
	nodes2contract={k:sorted(nodes2contract[k]) for k in nodes2contract.keys() if len(nodes2contract[k])>0}
	return claim_g,nodes2remove,nodes2contract

#fetch fred graph files from their API. slow and dependent on rate
def fredParse(claim_type,claims_path,claims,init,end,key,graph_type):
	errorclaimid=[]
	#fred starts
	start=time.time()
	start2=time.time()
	daysec=86400
	minsec=60
	rdf=rdflib.Graph()
	clean_claims={}
	os.makedirs(claims_path, exist_ok=True)
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
						claim_g,nodes2remove,nodes2contract=checkClaimGraph(g,claim_type,claim_ID,graph_type)
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
						claim_g=nx.relabel_nodes(claim_g,lambda x:nodelabel_mapper(x))
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

#Function to passively parse existing fetched fred rdf files
def passiveFredParse(index,claim_type,claims_path,claim_IDs,init,end,graph_type):
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
		claim_g,nodes2remove,nodes2contract=checkClaimGraph(g,claim_type,claim_ID,graph_type)
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
		claim_g=nx.relabel_nodes(claim_g,lambda x:nodelabel_mapper(x))
		nx.write_graphml(claim_g,filename+".graphml",prettyprint=True)
		#plot claim graph
		# plotFredGraph(claim_g,filename+".png")
	return index,errorclaimid,clean_claims

def fetchFred(rdf_path,graph_path,graph_type,fcg_label,init,passive,cpu):
	multigraph_types={'undirected':'nx.MultiGraph','directed':'nx.MultiDiGraph'}
	graph_types={'undirected':'nx.MultiGraph','directed':'nx.MultiDiGraph'}
	fcg_path=os.path.join(graph_path,"fred",fcg_label)
	#If union of tfcg and ffcg wants to be created i.e ufcg
	claim_types={"tfcg":"true","ffcg":"false"}
	claim_type=claim_types[fcg_label]
	claims_path=os.path.join(rdf_path,"{}_claims".format(claim_type))
	claim_IDs=list(np.load(os.path.join(rdf_path,"{}_claimID.npy".format(claim_type))))
	#parsing the graph for each claim
	#passive: if graph rdf files have already been fetched from fred. Faster parallelizable
	claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)),index_col=0)
	if passive:
		n=int(len(claim_IDs)/cpu)+1
		if cpu>1:
			pool=mp.Pool(processes=cpu)						
			results=[pool.apply_async(passiveFredParse, args=(index,claim_type,claims_path,claim_IDs,index*n,(index+1)*n,graph_types[graph_type])) for index in range(cpu)]
			output=sorted([p.get() for p in results],key=lambda x:x[0])
			errorclaimid=list(chain(*map(lambda x:x[1],output)))
			clean_claims=dict(ChainMap(*map(lambda x:x[2],output)))
		else:
			index,errorclaimid,clean_claims=passiveFredParse(0,claim_type,claims_path,claim_IDs,0,n,graph_types[graph_type])
	#if graph rdf files have not been fetched. Slower, dependent on rate limits
	else:
		
		keys={"tfcg":"Bearer a5c2a808-cc39-38e6-898d-84ab912b1e5d","ffcg":"Bearer 0d9d562e-a2aa-30df-90df-d52674f2e1f0"}
		end=len(claims)
		errorclaimid,clean_claims=fredParse(claim_type,claims_path,claims,init,end,keys[fcg_label],graph_types[graph_type])
	np.save(os.path.join(rdf_path,"{}_error_claimID.npy".format(fcg_label)),errorclaimid)
	with codecs.open(os.path.join(rdf_path,"{}claims_clean.json".format(claim_type)),"w","utf-8") as f:
		f.write(json.dumps(clean_claims,indent=4,ensure_ascii=False))

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Create fred graph')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/rdf_files/')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to store the graphs',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg','ffcg'],help='True or False FactCheckGraph')
	parser.add_argument('-i','--init', metavar='Index Start',type=int,help='Index number of claims to start from',default=0)
	parser.add_argument('-p','--passive',action='store_true',help='Passive or not',default=False)
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	parser.add_argument('-gt','--graphtype', metavar='Graph Type Directed/Undirected',type=str,choices=['directed','undirected'],default='undirected')
	args=parser.parse_args()
	fetchFred(args.rdfpath,args.graphpath,args.graphtype,args.fcgtype,args.init,args.passive,args.cpu)




