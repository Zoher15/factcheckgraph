import numpy as np
import pandas as pd
import rdflib
from rdflib import BNode, Literal
from rdflib.namespace import RDF
import networkx as nx 
import re
import pdb
import codecs
from itertools import combinations
import sys
import os 
import json
from IPython.core.debugger import set_trace

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
# set_trace()
# for f in trueclaims:
# 	g=rdflib.Graph()
# 	# filename="5005"+".rdf"
# 	filename="/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/"+str(f)+".rdf"

# for f in falseclaims:
# 	g=rdflib.Graph()
# 	# filename="5005"+".rdf"
# 	filename="/gpfs/home/z/k/zkachwal/Carbonate/RDF Files/"+str(f)+".rdf"
G=nx.Graph()
edges=[(0,1),(2,3)]
G.add_edges_from(edges)
nodes=[4,5,6,7]
G.add_nodes_from(nodes)
edgelist=np.asarray([[0,1,1],[2,3,1]])
np.save("test_edgelist.npy",edgelist)
with codecs.open("test_uris.txt","w","utf-8") as f:
	for node in G.nodes():
		f.write(str(node)+"\n")
comb=combinations(G.nodes,2)
combs=np.asarray(list(map(lambda x:[np.nan,x[0],np.nan,np.nan,x[1],np.nan,np.nan],comb)))
with codecs.open('test_pairs.txt',"w","utf-8") as f:
	for line in combs:
		f.write("{} {} {} {} {} {} {}\n".format(str(line[0]),str(int(line[1])),str(line[2]),str(line[3]),str(int(line[4])),str(line[5]),str(line[6])))
