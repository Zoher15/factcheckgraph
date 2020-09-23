import networkx as nx
import pandas as pd 
import numpy as np
import argparse
import rdflib
import re
import os
import codecs
import datetime
#Function to calculate stats of interest for a given graph
def calculate_stats(graph_path):
	fcg_types={"fred":["tfcg","ffcg","ufcg"],"fred1":["tfcg1","ffcg1","ufcg1"],"fred2":["tfcg2","ffcg2","ufcg2"],"fred3":["tfcg3","ffcg3","ufcg3"],"co_occur":["tfcg_co","ffcg_co","ufcg_co"],
	"backbone_df":["tfcg_bbdf","ffcg_bbdf","ufcg_bbdf"],"backbone_dc":["tfcg_bbdc","ffcg_bbdc","ufcg_bbdc"],
	"largest_ccf":["tfcg_lgccf","ffcg_lgccf","ufcg_lgccf"],"largest_ccc":["tfcg_lgccc","ffcg_lgccc","ufcg_lgccc"],
	"old_fred":["tfcg_old","ffcg_old","ufcg_old"]}
	df=pd.DataFrame()
	for fcg_class in fcg_types.keys():
		for fcg_label in fcg_types[fcg_class]:
			read_path=os.path.join(graph_path,fcg_class,fcg_label,"stats")
			if os.path.exists(read_path):
				temp_df=pd.read_csv(os.path.join(read_path,fcg_label+"_stats.csv"),index_col=0)
				df=pd.concat([df, temp_df], axis=1, sort=False)
	df = df.reindex(sorted(df.columns), axis=1)
	x = datetime.datetime.now().strftime("%c").replace(" ","_")
	df.to_csv(os.path.join(graph_path,"compiled_stats "+x+".csv"))

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='calculate stats for graphs')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/')
	args=parser.parse_args()
	calculate_stats(args.graphpath)