# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
import rdflib
from rdflib import BNode, Literal
from rdflib.namespace import RDF
from py2neo import Graph, NodeMatcher, RelationshipMatcher
from itertools import combinations
from sklearn import metrics
import codecs
import csv
import re
from decimal import Decimal
import networkx as nx 
import matplotlib
import scipy.stats as stats
matplotlib.use('TkAgg')
# font = {'family' : 'Normal',
#         'size'   : 12}
# matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pdb
import sys
import os 
import json
from IPython.core.debugger import set_trace
'''
The goal of this script is the following:
1. Read uris (save_uris) and triples (save_edgelist) from the Neo4j Database
2. Create files that feed into Knowledge Linker (code for calculating shortest paths in a knowledge graph)
3. Calculate Graph Statistics
3. Plot and Calculate ROC for the given triples versus randomly generated triples
'''
#Mode can be TFCG or FFCG
mode=sys.argv[1]
#PC can be 0 local, or 1 Carbonate
pc=int(sys.argv[2])
port={"FFCG":"7687","TFCG":"11007"}
g=rdflib.Graph()
graph = Graph("bolt://127.0.0.1:"+port[mode],password="1234")
#Getting the list of degrees of the given FactCheckGraph from Neo4j
def save_degree():
	tx = graph.begin()
	degreelist=np.asarray(list(map(lambda x:x['degree'],tx.run("Match (n)-[r]-(m) with n,count(m) as degree return degree"))))
	tx.commit()
	print(tx.finished())
	np.save(os.path.join(mode,mode+"_degreelist.npy"),degreelist)
	return degreelist

def save_edgelist(uris_dict):
	tx = graph.begin()
	graph_triples=tx.run("MATCH (n)-[r]-(m) return n,r,m;")
	tx.commit()
	print(tx.finished())
	triple_list=[]
	for triple in graph_triples:
		triple_list.append(triple)
	edgelist=[[uris_dict[triple_list[i]['n']['uri']],uris_dict[triple_list[i]['m']['uri']],1] for i in range(len(triple_list))]
	edgelist=np.asarray(edgelist)
	np.save(os.path.join(mode,mode+"_edgelist.npy"),edgelist)
	with codecs.open(os.path.join(mode,mode+'_edgelist.txt'),"w","utf-8") as f:
		for line in edgelist:
			f.write("{} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2])))
	return edgelist