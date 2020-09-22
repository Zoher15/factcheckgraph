# coding: utf-8
import codecs
import json
import numpy as np
from statistics import mean

def aggregate_weights(claim_D,mode,mode2):
	#mode can be w d or f
	#mode2 can be max, min, sum, mean
	edge_weights=[]
	for edge,evalues in claim_D.items():
		if type(evalues)!=list:
			#edge looks like this: "('db:John_McCain', 'db:United_States_Senate', 0.09, 1.25)"
			#u is the source node,v is the target node, w the special weight, d the 1/similarity
			u,v,w,d=eval(edge.replace("inf","np.inf"))
			# u,v,w,d,f=eval(edge.replace("inf","np.inf"))
			edge_weights.append(eval(mode))
	return eval("{}(edge_weights)".format(mode2))


with codecs.open("/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/graphs/fred/paths/tfcg_true_(roberta-base-nli-stsb-mean-tokens)/paths_directed_w.json","r","utf-8") as f:t1=json.loads(f.read())
with codecs.open("/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/graphs/fred/paths/tfcg_false_(roberta-base-nli-stsb-mean-tokens)/paths_directed_w.json","r","utf-8") as f:f1=json.loads(f.read())
with codecs.open("/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/graphs/fred/paths/tfcg_true_(roberta-base-nli-stsb-mean-tokens)/paths_undirected_w.json","r","utf-8") as f:t2=json.loads(f.read())
with codecs.open("/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/graphs/fred/paths/tfcg_false_(roberta-base-nli-stsb-mean-tokens)/paths_undirected_w.json","r","utf-8") as f:f2=json.loads(f.read())
d='w'
a='mean'
e='roberta-base-nli-stsb-mean-tokens'
t1s={int(t[0]):aggregate_weights(t[1],d,a) for t in t1.items()}
t2s={int(t[0]):aggregate_weights(t[1],d,a) for t in t2.items()}
f1s={int(t[0]):aggregate_weights(t[1],d,a) for t in f1.items()}
f2s={int(t[0]):aggregate_weights(t[1],d,a) for t in f2.items()}
t2s={k:t2s[k] for k in t1s.keys()}
f2s={k:f2s[k] for k in f1s.keys()}
t1t=[tuple([t[0],t[1]]) for t in t1s.items()]
t2t=[tuple([t[0],t[1]]) for t in t2s.items()]
f2t=[tuple([t[0],t[1]]) for t in f2s.items()]
f1t=[tuple([t[0],t[1]]) for t in f1s.items()]
len(t1t)
len(t2t)
len(f2t)
len(f1t)
from operator import itemgetter
t1t=sorted(t1t,key=itemgetter(0))
t2t=sorted(t2t,key=itemgetter(0))
f2t=sorted(f2t,key=itemgetter(0))
f1t=sorted(f1t,key=itemgetter(0))
tID1lt2=[t1t[i][0] for i in range(len(t1t)) if t1t[i][1]<t2t[i][1]]
tID1gt2=[t1t[i][0] for i in range(len(t1t)) if t1t[i][1]>t2t[i][1]]
fID1gt2=[f1t[i][0] for i in range(len(f1t)) if f1t[i][1]>f2t[i][1]]
fID1lt2=[f1t[i][0] for i in range(len(f1t)) if f1t[i][1]<f2t[i][1]]