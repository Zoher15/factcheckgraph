import os
import re
import sys
import time
import rdflib
import requests
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

__author__ = 'Misael Mongiovi, Andrea Giovanni Nuzzolese'
# Original script edited by Zoher Kachwala for FactCheckGraph

plugin.register('application/rdf+xml', Serializer, 'rdflib.plugins.serializers.rdfxml', 'XMLSerializer')
plugin.register('xml', Serializer, 'rdflib.plugins.serializers.rdfxml', 'XMLSerializer')

class FredType(Enum):
    Situation = 1
    Event = 2
    NamedEntity = 3
    SkolemizedEntity = 4
    Quality = 5
    Concept = 6

class NodeType(Enum):
    Class = 1
    Instance = 0

class ResourceType(Enum):
    Fred = 0
    Dbpedia = 1
    Verbnet = 2

class EdgeMotif(Enum):
    Identity = 1
    Type = 2
    SubClass = 3
    Equivalence = 4
    Role = 5
    Modality = 6
    Negation = 7
    Property = 8

class NaryMotif(Enum):
    Event = 1
    Situation = 2
    OtherEvent = 3
    Concept = 4

class PathMotif(Enum):
    Type = 1
    SubClass = 2

class ClusterMotif(Enum):
    Identity = 1
    Equivalence = 2
    IdentityEquivalence = 3 #all concepts tied by a sequence of sameAs and equivalentClass in any direction

class Role(Enum):
    Agent = 1
    Patient = 2
    Theme = 3
    Location = 4
    Time = 5
    Involve = 6
    Declared = 7
    VNOblique = 8
    LocOblique = 9
    ConjOblique = 10
    Extended = 11
    Associated = 12

class FredNode(object):
    def __init__(self,nodetype,fredtype,resourcetype):
        self.Type = nodetype
        self.FredType = fredtype
        self.ResourceType = resourcetype

class FredEdge(object):
    def __init__(self,edgetype):
        self.Type = edgetype


class FredGraph:
    def __init__(self,rdf):
        self.rdf = rdf

    def getNodes(self):
        nodes = set()
        for a, b, c in self.rdf:
            nodes.add(a.strip())
            nodes.add(c.strip())
        return nodes

    def getClassNodes(self):
        query = "PREFIX owl: <http://www.w3.org/2002/07/owl#> " \
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                "SELECT ?t WHERE { " \
                "?i a ?t1 . " \
                "?t1 (owl:equivalentClass | ^owl:equivalentClass | rdfs:sameAs | ^rdfs:sameAs | rdfs:subClassOf)* ?t }"

        nodes = set()
        res = self.rdf.query(query)
        for el in res:
            nodes.add(el[0].strip())
        return nodes

    def getInstanceNodes(self):
        nodes = self.getNodes()
        return nodes.difference(self.getClassNodes())

    def getEventNodes(self):
        query = "PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#> " \
                "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                "PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                "SELECT ?e WHERE { ?e a ?t . ?t rdfs:subClassOf* dul:Event }"

        nodes = set()
        res = self.rdf.query(query)
        for el in res:
            nodes.add(el[0].strip())
        return nodes

    def getSituationNodes(self):
        query = "PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#> " \
                "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                "PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                "SELECT ?e WHERE { ?e a ?t . ?t rdfs:subClassOf* boxing:Situation }"

        nodes = set()
        res = self.rdf.query(query)
        for el in res:
            nodes.add(el[0].strip())
        return nodes

    def getNamedEntityNodes(self):
        nodes = self.getNodes()
        events = self.getEventNodes()
        classes = self.getClassNodes()
        qualities = self.getQualityNodes()
        situation = self.getSituationNodes()

        ne = set()
        for n in nodes:
            if n not in classes and n not in qualities and n not in events and n not in situation:
                suffix = n[n.find("_", -1):]
                if suffix.isdigit() == False:
                    ne.add(n)
        return ne

    def getSkolemizedEntityNodes(self):
        nodes = self.getNodes()
        events = self.getEventNodes()
        classes = self.getClassNodes()
        qualities = self.getQualityNodes()
        situation = self.getSituationNodes()

        ne = set()
        for n in nodes:
            if n not in classes and n not in qualities and n not in events and n not in situation:
                suffix = n[n.find("_", -1):]
                if suffix.isdigit() == True:
                    ne.add(n)
        return ne

    def getQualityNodes(self):
        query = "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                "SELECT ?q WHERE { ?i dul:hasQuality ?q }"
        nodes = set()
        res = self.rdf.query(query)
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

        nodes = dict()
        query = "PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#> " \
                "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                "PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                "PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
                "SELECT ?n ?class ?x WHERE { { ?n a ?t . ?t rdfs:subClassOf* boxing:Situation bind (1 as ?x) bind (0 as ?class) } " \
                "UNION {?n a ?t . ?t rdfs:subClassOf* dul:Event bind (2 as ?x)  bind (0 as ?class)} " \
                "UNION {?i a ?t . ?t (owl:equivalentClass | ^owl:equivalentClass | rdfs:sameAs | ^rdfs:sameAs | rdfs:subClassOf)* ?n bind (6 as ?x) bind (1 as ?class)} }"

        res = self.rdf.query(query)
        for el in res:
            node = el[0].strip()
            cl = NodeType(el[1].value)
            type = FredType(el[2].value)
            nodes[node] = FredNode(cl,type,getResource(node))

        #if not an event nor situation nor class

        qualities = self.getQualityNodes()
        for n in qualities:
            if n not in nodes:
                nodes[n] = FredNode(NodeType.Instance,FredType.Quality,getResource(n))

        #if not even quality

        for n in self.getNodes():
            if n not in nodes:
                suffix = n[n.find("_", -1):]
                if n not in qualities and suffix.isdigit() == False:
                    nodes[n] = FredNode(NodeType.Instance,FredType.NamedEntity,getResource(n))
                else:
                    nodes[n] = FredNode(NodeType.Instance,FredType.SkolemizedEntity,getResource(n))

        return nodes

    def getEdges(self):
        return [(a.strip(),b.strip(),c.strip()) for (a,b,c) in self.rdf]

    #def getRoleEdges(self):
    #    return self.getEdgeMotif(EdgeMotif.Role)

    def getEdgeMotif(self,motif):
        if motif == EdgeMotif.Role:
            query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                    "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                    "SELECT ?i ?p ?o ?r WHERE " \
                    "{?i ?p ?o . ?i a ?t . ?t rdfs:subClassOf* dul:Event BIND (5 as ?r) }"
        elif motif == EdgeMotif.Identity:
            query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                    "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                    "PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
                    "SELECT ?i ?p ?o ?r WHERE " \
                    "{?i ?p ?o . FILTER(?p = owl:sameAs ) BIND (1 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}}"
        elif motif == EdgeMotif.Type:
            query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> " \
                    "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                    "SELECT ?i ?p ?o ?r WHERE " \
                    "{?i ?p ?o . FILTER(?p = rdf:type ) BIND (2 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}}"
        elif motif == EdgeMotif.SubClass:
            query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                    "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                    "SELECT ?i ?p ?o ?r WHERE " \
                    "{?i ?p ?o . FILTER(?p = rdfs:subClassOf ) BIND (3 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}}"
        elif motif == EdgeMotif.Equivalence:
            query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                    "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                    "PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
                    "SELECT ?i ?p ?o ?r WHERE " \
                    "{?i ?p ?o . FILTER(?p = owl:equivalentClass ) BIND (4 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}}"
        elif motif == EdgeMotif.Modality:
            query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                    "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                    "PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
                    "SELECT ?i ?p ?o ?r WHERE " \
                    "{?i ?p ?o . FILTER(?p = boxing:hasModality ) BIND (6 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}}"
        elif motif == EdgeMotif.Negation:
            query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                    "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                    "PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
                    "SELECT ?i ?p ?o ?r WHERE " \
                    "{?i ?p ?o . FILTER(?p = boxing:hasTruthValue ) BIND (7 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}}"
        elif motif == EdgeMotif.Property:
            query = "PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#> " \
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
        if motif == PathMotif.Type:
            query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> " \
                    "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                    "SELECT ?i ?o WHERE " \
                    "{?i rdf:type ?t . ?t rdfs:subClassOf* ?o}"
        elif motif == PathMotif.SubClass:
            query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                    "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                    "SELECT ?i ?o WHERE " \
                    "{?i rdfs:subClassOf+ ?o}"
        else:
            raise Exception("Unknown motif: " + str(motif))

        return [(el[0].strip(),el[1].strip()) for el in self.rdf.query(query)]

    def getClusterMotif(self,motif):
        if motif == ClusterMotif.Identity:
            query = "PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
                    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> " \
                    "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                    "SELECT DISTINCT ?s ?o WHERE " \
                    "{ ?s (owl:sameAs|^owl:sameAs)+ ?o } ORDER BY ?s "
        elif motif == ClusterMotif.Equivalence:
            query = "PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
                    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> " \
                    "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                    "SELECT DISTINCT ?s ?o WHERE " \
                    "{ ?s (^owl:equivalentClass|owl:equivalentClass)+ ?o } ORDER BY ?s "
        elif motif == ClusterMotif.IdentityEquivalence:
            query = "PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
                    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> " \
                    "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                    "SELECT DISTINCT ?s ?o WHERE " \
                    "{ ?s (owl:sameAs|^owl:sameAs|^owl:equivalentClass|owl:equivalentClass)+ ?o } ORDER BY ?s "
        else:
            raise Exception("Unknown motif: " + str(motif))

        results = self.rdf.query(query)

        clusters = list()
        used = set()
        olds = None
        currentset = set()
        for el in results:
            s = el[0].strip()
            o = el[1].strip()
            if s != olds:
                if len(currentset) != 0:
                    currentset.add(olds)
                    clusters.append(currentset)
                    used = used.union(currentset)
                    currentset = set()
                fillSet = False if s in used else True
            if fillSet == True:
                currentset.add(o)
            olds = s

        if len(currentset) != 0:
            currentset.add(olds)
            clusters.append(currentset)

        return clusters

    def getInfoEdges(self):
        edges = dict()
        query = "PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#> " \
                "PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> " \
                "PREFIX boxing: <http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#> " \
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                "PREFIX owl: <http://www.w3.org/2002/07/owl#>" \
                "" \
                "SELECT ?i ?p ?o ?r WHERE {" \
                "{?i ?p ?o . ?i a ?t . ?t rdfs:subClassOf* dul:Event BIND (5 as ?r) }" \
                "UNION" \
                "{?i ?p ?o . FILTER(?p = owl:sameAs ) BIND (1 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}  }" \
                "UNION" \
                "{?i ?p ?o . FILTER(?p = rdf:type ) BIND (2 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}  }" \
                "UNION" \
                "{?i ?p ?o . FILTER(?p = rdfs:subClassOf ) BIND (3 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}  }" \
                "UNION" \
                "{?i ?p ?o . FILTER(?p = owl:equivalentClass ) BIND (4 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}  }" \
                "UNION" \
                "{?i ?p ?o . FILTER(?p = boxing:hasModality ) BIND (6 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}  }" \
                "UNION" \
                "{?i ?p ?o . FILTER(?p = boxing:hasTruthValue ) BIND (7 as ?r) FILTER NOT EXISTS {?i a ?t . ?t rdfs:subClassOf* dul:Event}  }" \
                "}"

        res = self.rdf.query(query)
        for el in res:
            edges[(el[0].strip(),el[1].strip(),el[2].strip())] = FredEdge(EdgeMotif(el[3].value))
        for e in self.getEdges():
            if e not in edges:
                edges[e] = FredEdge(EdgeMotif.Property)
        return edges

    def getNaryMotif(self,motif):
        def fillRoles(el):
            relations = dict()
            if el['agent'] != None:
                relations[Role.Agent] = el['agent']
            if el['patient'] != None:
                relations[Role.Patient] = el['patient']
            if el['theme'] != None:
                relations[Role.Theme] = el['theme']
            if el['location'] != None:
                relations[Role.Theme] = el['location']
            if el['time'] != None:
                relations[Role.Theme] = el['time']
            if el['involve'] != None:
                relations[Role.Theme] = el['involve']
            if el['declared'] != None:
                relations[Role.Theme] = el['declared']
            if el['vnoblique'] != None:
                relations[Role.Theme] = el['vnoblique']
            if el['locoblique'] != None:
                relations[Role.Theme] = el['locoblique']
            if el['conjoblique'] != None:
                relations[Role.Theme] = el['conjoblique']
            if el['extended'] != None:
                relations[Role.Theme] = el['extended']
            if el['associated'] != None:
                relations[Role.Theme] = el['associated']
            return relations

        query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
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
                " FILTER (?agentiverole = vnrole:Agent || ?agentiverole = vnrole:Actor1 || ?agentiverole = vnrole:Actor2 || ?agentiverole = vnrole:Experiencer || ?agentiverole = vnrole:Cause || ?agentiverole = boxer:agent)}" \
                "OPTIONAL " \
                " {?node ?passiverole ?patient" \
                " FILTER (?passiverole = vnrole:Patient || ?passiverole = vnrole:Patient1 || ?passiverole = vnrole:Patient2 || ?passiverole = vnrole:Beneficiary || ?passiverole = boxer:patient || ?passiverole = vnrole:Recipient || ?passiverole = boxer:recipient)} " \
                "OPTIONAL " \
                " {?node ?themerole ?theme" \
                " FILTER (?themerole = vnrole:Theme || ?themerole = vnrole:Theme1 || ?themerole = vnrole:Theme2 || ?themerole = boxer:theme)} " \
                "OPTIONAL " \
                " {?node ?locativerole ?location" \
                " FILTER (?locativerole = vnrole:Location || ?locativerole = vnrole:Destination || ?locativerole = vnrole:Source || ?locativerole = fred:locatedIn)} " \
                "OPTIONAL " \
                " {?node ?temporalrole ?time" \
                " FILTER (?temporalrole = vnrole:Time)} " \
                "OPTIONAL " \
                " {?node ?situationrole ?involve" \
                " FILTER (?situationrole = boxing:involves)} " \
                "OPTIONAL " \
                " {?node ?declarationrole ?declared" \
                " FILTER (?declarationrole = boxing:declaration || ?declarationrole = vnrole:Predicate || ?declarationrole = vnrole:Proposition)} " \
                "OPTIONAL " \
                " { ?node ?vnobrole ?vnoblique " \
                " FILTER (?vnobrole = vnrole:Asset || ?vnobrole = vnrole:Attribute || ?vnobrole = vnrole:Extent || ?vnobrole = vnrole:Instrument || ?vnobrole = vnrole:Material || ?vnobrole = vnrole:Oblique || ?vnobrole = vnrole:Oblique1 || ?vnobrole = vnrole:Oblique2 || ?vnobrole = vnrole:Product || ?vnobrole = vnrole:Stimulus || ?vnobrole = vnrole:Topic || ?vnobrole = vnrole:Value)}" \
                "OPTIONAL " \
                " {?node ?preposition ?locoblique" \
                " FILTER (?preposition = fred:about || ?preposition = fred:after || ?preposition = fred:against || ?preposition = fred:among || ?preposition = fred:at || ?preposition = fred:before || ?preposition = fred:between || ?preposition = fred:by || ?preposition = fred:concerning || ?preposition = fred:for || ?preposition = fred:from || ?preposition = fred:in || ?preposition = fred:in_between || ?preposition = fred:into || ?preposition = fred:of || ?preposition = fred:off || ?preposition = fred:on || ?preposition = fred:onto || ?preposition = fred:out_of || ?preposition = fred:over || ?preposition = fred:regarding || ?preposition = fred:respecting || ?preposition = fred:through || ?preposition = fred:to || ?preposition = fred:towards || ?preposition = fred:under || ?preposition = fred:until || ?preposition = fred:upon || ?preposition = fred:with)}" \
                "OPTIONAL " \
                " {{?node ?conjunctive ?conjoblique" \
                " FILTER (?conjunctive = fred:as || ?conjunctive = fred:when || ?conjunctive = fred:after || ?conjunctive = fred:where || ?conjunctive = fred:whenever || ?conjunctive = fred:wherever || ?conjunctive = fred:because || ?conjunctive = fred:if || ?conjunctive = fred:before || ?conjunctive = fred:since || ?conjunctive = fred:unless || ?conjunctive = fred:until || ?conjunctive = fred:while)} UNION {?conjoblique ?conjunctive ?node FILTER (?conjunctive = fred:once || ?conjunctive = fred:though || ?conjunctive = fred:although)}}" \
                "OPTIONAL " \
                " {?node ?periphrastic ?extended" \
                " FILTER (?periphrastic != ?vnobrole && ?periphrastic != ?preposition && ?periphrastic != ?conjunctive && ?periphrastic != ?agentiverole && ?periphrastic != ?passiverole && ?periphrastic != ?themerole && ?periphrastic != ?locativerole && ?periphrastic != ?temporalrole && ?periphrastic != ?situationrole && ?periphrastic != ?declarationrole && ?periphrastic != ?associationrole && ?periphrastic != boxing:hasTruthValue && ?periphrastic != boxing:hasModality && ?periphrastic != dul:hasQuality && ?periphrastic != dul:associatedWith && ?periphrastic != dul:hasRole &&?periphrastic != rdf:type)}" \
                "OPTIONAL " \
                " {?node ?associationrole ?associated" \
                " FILTER (?associationrole = boxer:rel || ?associationrole = dul:associatedWith)} " \
                "}" \
                " ORDER BY ?type"

        results = self.rdf.query(query)
        motifocc = dict()
        for el in results:
            if NaryMotif(el['type']) == motif:
                motifocc[el['node'].strip()] = fillRoles(el)

        return motifocc

    def getCompactGraph(self):
        pass

def preprocessText(text):
    nt = text.replace("-"," ")
    nt = nt.replace("#"," ")
    nt = nt.replace(chr(96),"'") #`->'
    nt = nt.replace("'nt "," not ")
    nt = nt.replace("'ve "," have ")
    nt = nt.replace(" what's "," what is ")
    nt = nt.replace("What's ","What is ")
    nt = nt.replace(" where's "," where is ")
    nt = nt.replace("Where's ","Where is ")
    nt = nt.replace(" how's "," how is ")
    nt = nt.replace("How's ","How is ")
    nt = nt.replace(" he's "," he is ")
    nt = nt.replace(" she's "," she is ")
    nt = nt.replace(" it's "," it is ")
    nt = nt.replace("He's ","He is ")
    nt = nt.replace("She's ","She is ")
    nt = nt.replace("It's ","It is ")
    nt = nt.replace("'d "," had ")
    nt = nt.replace("'ll "," will ")
    nt = nt.replace("'m "," am ")
    nt = nt.replace(" ma'am "," madam ")
    nt = nt.replace(" o'clock "," of the clock ")
    nt = nt.replace(" 're "," are ")
    nt = nt.replace(" y'all "," you all ")

    nt = nt.strip()
    if nt[len(nt)-1]!='.':
        nt = nt + "."

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

def fredParse(data,claim_type,init):
    graph_type={"True":"TFCG","False":"FFCG"}
    FCG_label=graph_type[claim_type]
    key="Bearer 56a28f54-7918-3fdd-9d6f-850f13bd4041"
    errorclaimid=[]
    #fred starts
    start=time.time()
    start2=time.time()
    daysec=86400
    minsec=60
    FCG=nx.Graph()
    rdf = rdflib.Graph()
    FCG_filterdata={}
    #Read stuff
    try:
        rdf.parse('/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/{} Claims/{}claims.rdf'.format(claim_type,claim_type), format='application/rdf+xml')
        with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/{} Claims/{}Claims_filterdata.json".format(claim_type,claim_type),"w","utf-8") as f:
            FCG_filterdata=json.loads(f.read())
        FCG=nx.read_edgelist(os.path.join(FCG_label,FCG_label+".edgelist"))
    except:
        pass
    for i in range(init,len(data)):
        dif=abs(time.time()-start)
        diff=abs(daysec-dif)
        while True:
            try:
                dif=abs(time.time()-start)
                dif2=abs(time.time()-start2)
                diff=abs(daysec-dif)
                claimID=data.iloc[i]['claimID']
                sentence=html.unescape(data.iloc[i]['claim_text']).replace("`","'")
                print("Index:",i,"Claim ID:",claimID," DayLim2Go:",round(diff),"MinLim2Go:",round(min(abs(minsec-dif2),60)))
                filename="/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/{} Claims/Claim{}.rdf".format(claim_type,str(claimID))
                r=getFredGraph(preprocessText(sentence),key,filename)
                if "You have exceeded your quota" not in r.text and "Runtime Error" not in r.text and "Service Unavailable" not in r.text:
                    if r.status_code in range(100,500) and r.text:
                        g=openFredGraph(filename)
                        nx_graph,removed_edges,contracted_edges=checkFredGraph(g)
                        plotFredGraph(nx_graph,filename)
                        FCG=nx.compose(FCG,nx_graph)
                        FCG_filterdata[str(claimID)]={}
                        FCG_filterdata[str(claimID)]['removed_edges']=removed_edges
                        FCG_filterdata[str(claimID)]['contracted_edges']=contracted_edges
                        rdf.parse(filename,format='application/rdf+xml')
                        rdf.serialize(destination='/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/{} Claims/{}claims.rdf'.format(claim_type,claim_type), format='application/rdf+xml')
                        with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/{} Claims/{}Claims_filterdata.json".format(claim_type,claim_type),"w","utf-8") as f:
                            f.write(json.dumps(FCG_filterdata,ensure_ascii=False))
                        nx.write_edgelist(FCG,os.path.join(FCG_label,FCG_label+".edgelist"))
                    else:
                        errorclaimid.append(filename.split("/")[-1].strip(".rdf"))
                        np.save("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/Error500_claimID.npy",errorclaimid)
                    break
                else:
                    diff2=min(abs(minsec-dif2),60)
                    print("Sleeping for ",round(diff2))
                    time.sleep(abs(diff2))
                    start2=time.time()
            except xml.sax._exceptions.SAXParseException:
                print("Exception Occurred")
                errorclaimid.append(claimID)
                break
    rdf.serialize(destination='/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/{} Claims/{}claims.rdf'.format(claim_type,claim_type), format='application/rdf+xml')
    with codecs.open("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/{} Claims/{}Claims_filterdata.json".format(claim_type,claim_type),"w","utf-8") as f:
        f.write(json.dumps(FCG_filterdata,ensure_ascii=False))
    nx.write_edgelist(FCG,os.path.join(FCG_label,FCG_label+".edgelist"))

def create_fred_network(init):
    data=pd.read_csv("/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/claimreviews_db2.csv",index_col=0)
    ##Dropping non-str rows
    filter=list(map(lambda x:type(x)!=str,data['rating_name']))
    data.drop(data[filter].index,inplace=True)
    print(data.groupby('fact_checkerID').count())
    trueregex=re.compile(r'(?i)^true|^correct$|^mostly true$|^geppetto checkmark$')
    falseregex=re.compile(r'(?i)^false|^mostly false|^pants on fire$|^four pinocchios$|^no\ |^no:|^distorts the facts|^wrong$')
    trueind=data['rating_name'].apply(lambda x:trueregex.match(x)!=None)
    trueclaims=data.loc[trueind]
    # set_trace()
    falseind=data['rating_name'].apply(lambda x:falseregex.match(x)!=None)
    falseclaims=data.loc[falseind]
    fredParse(trueclaims,"True",init)
    fredParse(falseclaims,"False",init)

def openFredGraph(filename):
    rdf = rdflib.Graph()
    rdf.parse(filename)
    return FredGraph(rdf)

def checkFredFile(filename):
    g = openFredGraph(filename)
    checkFredGraph(g)

def checkFredGraph(g):
    nx_graph=nx.Graph()
    removed_edges=[]
    contracted_edges=[]
    # print("getNodes")
    # for n in g.getNodes():
    #     print(n)

    # print("getClassNodes")
    # for n in g.getClassNodes():
    #     print(n)

    # print("getInstanceNodes")
    # for n in g.getInstanceNodes():
    #     print(n)

    # print("getEventNodes")
    # for n in g.getEventNodes():
    #     print(n)

    # print("getSituationNodes")
    # for n in g.getSituationNodes():
    #     print(n)

    # print("getNamedEntityNodes")
    # for n in g.getNamedEntityNodes():
    #     print(n)

    # print("getQualityNodes")
    # for n in g.getQualityNodes():
    #     print(n)

    # print("getInfoNodes")
    # ns = g.getInfoNodes()
    # for n in ns:
    #     print(n, ns[n].Type, ns[n].FredType, ns[n].ResourceType)
    regex_27=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#%27.*')
    regex_vndata=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/vn\/data\/(.*)')
    regex_freddata_low=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#([a-z]*)_.*')
    regex_freddata_upp=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/domain\.owl#([a-zA-Z]*)$')
    regex_dbpedia=re.compile(r'^http:\/\/dbpedia\.org\/resource\/(.*)')
    regex_dul=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/dul\/DUL\.owl#(.*)')
    regex_owl=re.compile(r'^http:\/\/www\.w3\.org\/2002\/07\/owl#(.*)')
    regex_schema=re.compile(r'^http:\/\/schema\.org\/(.*)')
    regex_quant=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/fred\/quantifiers\.owl#.*')
    regex_assoc=re.compile(r'^http:\/\/www\.ontologydesignpatterns\.org\/ont\/dul\/DUL\.owl#associatedWith$')
    # print("getEdges")
    for (a,b,c) in g.getEdges():
        if b=="http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#associatedWith":
            if regex_freddata_low.match(a) and regex_freddata_low.match(c):
                if regex_freddata_low.match(a)[1]==regex_freddata_low.match(c)[1]:
                    removed_edges.append((a,b,c))
            elif regex_freddata_low.match(a) and regex_freddata_upp.match(c):
                if regex_freddata_low.match(a)[1]==regex_freddata_upp.match(c)[1].lower():
                    removed_edges.append((a,b,c))
            elif regex_freddata_low.match(c) and regex_freddata_upp.match(a):
                if regex_freddata_low.match(c)[1]==regex_freddata_upp.match(a)[1].lower():
                    removed_edges.append((a,b,c))
        elif regex_quant.match(b) or regex_27.match(a) or regex_27.match(c) or regex_dul.match(a) or regex_dul.match(c):
            removed_edges.append((a,b,c))
        else:
            nx_graph.add_edge(a,c,label=b)
        # print(a,b,c)
    # print("getEdgeMotif(EdgeMotif.Role)")
    # for (a,b,c) in g.getEdgeMotif(EdgeMotif.Role):
    #     print(a,b,c)

    # print("getEdgeMotif(EdgeMotif.Identity)")
    # for (a,b,c) in g.getEdgeMotif(EdgeMotif.Identity):
    #     print(a,b,c)

    # print("getEdgeMotif(EdgeMotif.Type)")
    # for (a,b,c) in g.getEdgeMotif(EdgeMotif.Type):
    #     print(a,b,c)

    # print("getEdgeMotif(EdgeMotif.Property)")
    # for (a,b,c) in g.getEdgeMotif(EdgeMotif.Property):
    #     print(a,b,c)

    # print("getPathMotif(PathMotif.Type)")
    for (a,b) in g.getPathMotif(PathMotif.Type):
        if nx_graph.has_edge(a,b):
            if regex_freddata_low.match(a)!=None and regex_freddata_upp.match(b)!=None:
                if regex_freddata_low.match(a)[1]==regex_freddata_upp.match(b)[1].lower():
                    nx_graph = nx.contracted_edge(nx_graph,(b, a),self_loops=False)
                    contracted_edges.append((b,a))
            elif regex_freddata_low.match(b)!=None and regex_freddata_upp.match(a)!=None:
                if regex_freddata_low.match(b)[1]==regex_freddata_upp.match(a)[1].lower():
                    nx_graph = nx.contracted_edge(nx_graph,(a, b),self_loops=False)
                    contracted_edges.append((a,b))
            elif regex_freddata_low.match(a)!=None and regex_owl.match(b)!=None:
                if regex_freddata_low.match(a)[1]==regex_owl.match(b)[1].lower():
                    nx_graph = nx.contracted_edge(nx_graph,(b, a),self_loops=False)
                    contracted_edges.append((b,a))
            elif regex_freddata_low.match(b)!=None and regex_owl.match(a)!=None:
                if regex_freddata_low.match(b)[1]==regex_owl.match(a)[1].lower():
                    nx_graph = nx.contracted_edge(nx_graph,(a, b),self_loops=False)
                    contracted_edges.append((a,b))
            elif regex_schema.match(a) or regex_schema.match(b):
                removed_edges.append((a,b))
                nx_graph.remove_edge(a,b)

    # print("getEdgeMotif(EdgeMotif.SubClass)")
    # for (a,b,c) in g.getEdgeMotif(EdgeMotif.SubClass):
    #     print(a,b,c)

    # print("getEdgeMotif(EdgeMotif.Equivalence)")
    # for (a,b,c) in g.getEdgeMotif(EdgeMotif.Equivalence):
    #     print(a,b,c)

    # print("getEdgeMotif(EdgeMotif.Modality)")
    # for (a,b,c) in g.getEdgeMotif(EdgeMotif.Modality):
    #     print(a,b,c)

    # print("getEdgeMotif(EdgeMotif.Negation)")
    # for (a,b,c) in g.getEdgeMotif(EdgeMotif.Negation):
    #     print(a,b,c)

    # print("getInfoEdges")
    # es = g.getInfoEdges()
    # for e in es:
    #     print(e, es[e].Type)

    # print("getPathMotif(PathMotif.SubClass)")
    # for (a,b) in g.getPathMotif(PathMotif.SubClass):
    #     print(a,b)

    # print("getClusterMotif(ClusterMotif.Identity)")
    # for cluster in g.getClusterMotif(ClusterMotif.Identity):
    #     print(cluster)

    # print("getClusterMotif(ClusterMotif.Equivalence)")
    # for cluster in g.getClusterMotif(ClusterMotif.Equivalence):
    #     print(cluster)

    # print("getClusterMotif(ClusterMotif.IdentityEquivalence)")
    for cluster in g.getClusterMotif(ClusterMotif.IdentityEquivalence):
        # print(cluster)
        a=list(cluster)[0]
        b=list(cluster)[1]
        if nx_graph.has_edge(a,b):
            if regex_vndata.match(a) or regex_dbpedia.match(a):
                nx_graph = nx.contracted_edge(nx_graph,(a, b),self_loops=False)
                contracted_edges.append((a,b))
            elif regex_vndata.match(b) or regex_dbpedia.match(b):
                nx_graph = nx.contracted_edge(nx_graph,(b, a),self_loops=False)
                contracted_edges.append((a,b))

    # print("g.getNaryMotif(NaryMotif.Event)")
    # motif_occurrences = g.getNaryMotif(NaryMotif.Event)
    # for event in motif_occurrences:
    #     roles = motif_occurrences[event]
    #     print(event,"{", end=' ')
    #     for r in roles:
    #         print(r,":",roles[r],";", end=' ')
    #     print("}")

    # print("g.getNaryMotif(NaryMotif.Situation)")
    # motif_occurrences = g.getNaryMotif(NaryMotif.Situation)
    # for situation in motif_occurrences:
    #     roles = motif_occurrences[situation]
    #     print(event,"{", end=' ')
    #     for r in roles:
    #         print(r,":",roles[r],";", end=' ')
    #     print("}")

    # print("g.getNaryMotif(NaryMotif.OtherEvent)")
    # motif_occurrences = g.getNaryMotif(NaryMotif.OtherEvent)
    # for other_event in motif_occurrences:
    #     roles = motif_occurrences[other_event]
    #     print(event,"{", end=' ')
    #     for r in roles:
    #         print(r,":",roles[r],";", end=' ')
    #     print("}")

    # print("g.getNaryMotif(NaryMotif.Concept)")
    # motif_occurrences = g.getNaryMotif(NaryMotif.Concept)
    # for concept in motif_occurrences:
    #     roles = motif_occurrences[concept]
    #     print(event,"{", end=' ')
    #     for r in roles:
    #         print(r,":",roles[r],";", end=' ')
    #     print("}")
    nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))
    return nx_graph,removed_edges,contracted_edges

def plotFredGraph(nx_graph,filename):
    plt.figure()
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph,pos,labels={node:node.split("/")[-1].split("#")[-1] for node in nx_graph.nodes()},node_size=400)
    edge_labels = {(edge[0], edge[1]): edge[2]['label'].split("/")[-1].split("#")[-1] for edge in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph,pos,edge_labels)
    plt.axis('off')
    plt.savefig(filename.replace(".rdf",".png"))
    plt.close()
    plt.clf()