import pandas as pd 
import numpy as np
import argparse
import re
import os

def separate_claims(file,rdf_path):
	data=pd.read_csv(os.path.join(rdf_path,file))
	##Dropping non-english rows
	data=data[data['claim_lang']=='en']
	##Dropping non-str rows
	filter=list(map(lambda x:type(x)!=str,data['rating_name']))
	data.drop(data[filter].index,inplace=True)
	#drop duplicates with the same claimID
	import pdb
	pdb.set_trace()
	data.drop_duplicates('claimID',keep='first',inplace=True)
	print(data.groupby('factchecker').count())
	true_regex=re.compile(r'(?i)^true|^correct$|^mostly true$|^geppetto checkmark$')
	false_regex=re.compile(r'(?i)^false|^mostly false|^pants on fire$|^four pinocchios$|^no\ |^no:|^distorts the facts|^wrong$')
	true_ind=data['rating_name'].apply(lambda x:true_regex.match(x)!=None)
	true_claims=data.loc[true_ind].reset_index(drop=True)
	false_ind=data['rating_name'].apply(lambda x:false_regex.match(x)!=None)
	false_claims=data.loc[false_ind].reset_index(drop=True)
	import pdb
	pdb.set_trace()
	np.save(os.path.join(rdf_path,"true_claimID.npy"),list(true_claims["claimID"]))
	np.save(os.path.join(rdf_path,"false_claimID.npy"),list(false_claims["claimID"]))
	true_claims.to_csv(os.path.join(rdf_path,"true_claims.csv"))
	false_claims.to_csv(os.path.join(rdf_path,"false_claims.csv"))
	os.makedirs(os.path.join(rdf_path,"true_claims"), exist_ok=True)
	os.makedirs(os.path.join(rdf_path,"false_claims"), exist_ok=True)

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Separate claims into True and False')
	parser.add_argument('-f','--file', metavar='file name',type=str,help='Name of the csv file that stores the claim data',default='all_claims26mar2020.csv')
	parser.add_argument('-rdf','--rdfpath', metavar='rdf path',type=str,help='RDF path to read the file and store the new files',default='/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/covid19_rdf_files')
	args=parser.parse_args()
	separate_claims(args.file,args.rdfpath)