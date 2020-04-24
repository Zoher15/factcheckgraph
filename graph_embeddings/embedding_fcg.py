import matplotlib.pyplot as plt
import os
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
from time import time
# from gem.embedding.gf       import GraphFactorization
# from gem.embedding.hope     import HOPE
# from gem.embedding.lap      import LaplacianEigenmaps
# from gem.embedding.lle      import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne     import SDNE

def embedFred(graph_path,fcg_label,cpu,compilefred):
	# File that contains the edges. Format: source target
	# Optionally, you can add weights as third column: source target weight
	edge_f = os.path.join(graph_path,"fred"+str(compilefred),"tfcg"+str(compilefred),str(fcg_label)+"{}.edgelist".format(str(compilefred)))
	# Specify whether the edges are directed
	isDirected = True
	# Load graph
	G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
	G = G.to_directed()

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Embed fred graph')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Graph directory to retrieve the graphs from',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	parser.add_argument('-ft','--fcgtype', metavar='FactCheckGraph type',type=str,choices=['tfcg','ffcg','ufcg'],help='True False or Union FactCheckGraph')
	parser.add_argument('-cpu','--cpu',metavar='Number of CPUs',type=int,help='Number of CPUs available',default=1)
	parser.add_argument('-cf','--compilefred',metavar='Compile method #',type=int,help='Number of compile method',default=0)
	args=parser.parse_args()
	embedFred(args.graphpath,args.fcgtype,args.cpu,args.compilefred)