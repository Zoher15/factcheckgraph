import pandas as pd 
import numpy as np
import argparse
import re
import os

def separate_claims(file,path):
	data=pd.read_csv(os.path.join(path,file),index_col=0)
	##Dropping non-str rows
	filter=list(map(lambda x:type(x)!=str,data['rating_name']))
	data.drop(data[filter].index,inplace=True)
	print(data.groupby('fact_checkerID').count())
	true_regex=re.compile(r'(?i)^true|^correct$|^mostly true$|^geppetto checkmark$')
	false_regex=re.compile(r'(?i)^false|^mostly false|^pants on fire$|^four pinocchios$|^no\ |^no:|^distorts the facts|^wrong$')
	true_ind=data['rating_name'].apply(lambda x:true_regex.match(x)!=None)
	true_claims=data.loc[true_ind]
	false_ind=data['rating_name'].apply(lambda x:false_regex.match(x)!=None)
	false_claims=data.loc[false_ind]
	np.save(os.path.join(path,"true_claimID.npy"),list(true_claims["claimID"]))
	np.save(os.path.join(path,"false_claimID.npy"),list(false_claims["claimID"]))
	true_claims.to_csv(os.path.join(path,"true_claims.csv"),index=False)
	false_claims.to_csv(os.path.join(path,"false_claims.csv"),index=False)

if __name__== "__main__":
	parser.add_argument('-f','--file', metavar='file name',type=str,help='Name of the csv file that stores the claim data')
	parser.add_argument('-p','--path', metavar='file path',type=str,help='Path to read the file and store the new files')
	args=parser.parse_args()
	separate_claims(args.file,arg.path)