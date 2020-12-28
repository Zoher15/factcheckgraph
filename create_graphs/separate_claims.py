import pandas as pd 
import numpy as np
import argparse
import re
import os

def filterclaim(row):
	row['claim_text']=row['claim_text'].replace("Says that ","")
	row['claim_text']=row['claim_text'].replace("Says ","")
	row['claim_text']=row['claim_text'].replace("breaking: ","")
	row['claim_text']=row['claim_text'].replace("Breaking: ","")
	row['claim_text']=row['claim_text'].replace("BREAKING: ","")
	row['claim_text']=row['claim_text'].replace("Said that ","")
	row['claim_text']=row['claim_text'].replace("Said ","")
	row['claim_text']=row['claim_text'].replace("Claim ","")
	row['claim_text']=row['claim_text'].replace("Claim :","")
	firstletter=re.search(r'[a-zA-Z]',row['claim_text']).span()[0]
	row['claim_text']=row['claim_text'][0:firstletter]+row['claim_text'][firstletter].capitalize()+row['claim_text'][firstletter+1:]
	return row

def separate_claims(file,rdf_path):
	data=pd.read_csv(os.path.join(rdf_path,file),index_col=0)
	##Dropping non-str rows
	filter=list(map(lambda x:type(x)!=str,data['rating_name']))
	data.drop(data[filter].index,inplace=True)
	#drop duplicates with the same claimID
	data.drop_duplicates('claimID',keep='first',inplace=True)
	data=data.apply(filterclaim,axis=1)
	data.to_csv(os.path.join(rdf_path,"claims_dataset.csv"))
	print(data.groupby('fact_checkerID').count())
	true_regex=re.compile(r'(?i)^true|^correct$|^mostly true$|^geppetto checkmark$')
	false_regex=re.compile(r'(?i)^false|^mostly false|^pants on fire$|^four pinocchios$|^no\ |^no:|^distorts the facts|^wrong$')
	true_ind=data['rating_name'].apply(lambda x:true_regex.match(x)!=None)
	true_claims=data.loc[true_ind].reset_index(drop=True)
	false_ind=data['rating_name'].apply(lambda x:false_regex.match(x)!=None)
	false_claims=data.loc[false_ind].reset_index(drop=True)
	np.save(os.path.join(rdf_path,"true_claimID.npy"),list(true_claims["claimID"]))
	np.save(os.path.join(rdf_path,"false_claimID.npy"),list(false_claims["claimID"]))
	true_claims.to_csv(os.path.join(rdf_path,"true_claims.csv"))
	false_claims.to_csv(os.path.join(rdf_path,"false_claims.csv"))
	os.makedirs(os.path.join(rdf_path,"true_claims"), exist_ok=True)
	os.makedirs(os.path.join(rdf_path,"false_claims"), exist_ok=True)

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Separate claims into True and False')
	parser.add_argument('-f','--file', metavar='file name',type=str,help='Name of the csv file that stores the claim data',default='claims_dataset.csv')
	parser.add_argument('-rdf','--rdfpath', metavar='rdf path',type=str,help='RDF path to read the file and store the new files',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/rdf_files')
	args=parser.parse_args()
	separate_claims(args.file,args.rdfpath)