import pandas as pd 
import numpy as np
import argparse
import rdflib
from rdflib.compare import isomorphic,to_isomorphic
import re
import os

def separate_claims(file,rdf_path):
	data=pd.read_csv(os.path.join(rdf_path,file),index_col=0)
	##Dropping non-str rows
	filter=list(map(lambda x:type(x)!=str,data['rating_name']))
	data.drop(data[filter].index,inplace=True)
	#drop duplicates with the same claimID
	data.drop_duplicates('claimID',keep='first',inplace=True)
	print(data.groupby('fact_checkerID').count())
	true_regex=re.compile(r'(?i)^true|^correct$|^mostly true$|^geppetto checkmark$')
	false_regex=re.compile(r'(?i)^false|^mostly false|^pants on fire$|^four pinocchios$|^no\ |^no:|^distorts the facts|^wrong$')
	true_ind=data['rating_name'].apply(lambda x:true_regex.match(x)!=None)
	false_ind=data['rating_name'].apply(lambda x:false_regex.match(x)!=None)
	true_claimIDs=list(data.loc[true_ind].reset_index(drop=True)["claimID"])
	false_claimIDs=list(data.loc[false_ind].reset_index(drop=True)["claimID"])
	# np.save(os.path.join(rdf_path,"true_claimID2.npy"),list(true_claims["claimID"]))
	# np.save(os.path.join(rdf_path,"false_claimID2.npy"),list(false_claims["claimID"]))
	# true_claims.to_csv(os.path.join(rdf_path,"true_claims2.csv"))
	# false_claims.to_csv(os.path.join(rdf_path,"false_claims2.csv"))
	# import pdb
	# pdb.set_trace()
	# os.makedirs(os.path.join(rdf_path,"true_claims"), exist_ok=True)
	# os.makedirs(os.path.join(rdf_path,"false_claims"), exist_ok=True)
	return true_claimIDs,false_claimIDs

def count_claims(claims_path1,claims_path2,claim_IDs):
	#Reading claim_IDs
	errorclaimid1=[]
	errorclaimid2=[]
	iso=[]
	to_iso=[]
	match_fail=[]
	parse_fail=[]
	parse1_fail=[]
	parse2_fail=[]
	rdf1=rdflib.Graph()
	rdf2=rdflib.Graph()
	for i in range(len(claim_IDs)):
		parse_flag1=0
		parse_flag2=0
		match_flag1=0
		match_flag2=0
		claim_ID=claim_IDs[i]
		filename1=os.path.join(claims_path1,"{}".format(str(claim_ID)))
		filename2=os.path.join(claims_path2,"claim{}".format(str(claim_ID)))
		try:
			rdf1=rdflib.Graph()
			rdf1.parse(filename1+'.rdf')
		except:
			parse_flag1=1
			# print("Exception Occurred")
			errorclaimid1.append(claim_ID)
		try:
			rdf2=rdflib.Graph()
			rdf2.parse(filename2+'.rdf')
		except:
			parse_flag2=1
			# print("Exception Occurred")
			errorclaimid2.append(claim_ID)
		if parse_flag1 and parse_flag2:
			parse_fail.append(claim_ID)
		elif parse_flag1:
			parse1_fail.append(claim_ID)
		elif parse_flag2:
			parse2_fail.append(claim_ID)
		else:
			if isomorphic(rdf1,rdf2):
				iso.append(claim_ID)
			else:
				match_flag1=1
			if to_isomorphic(rdf1)==to_isomorphic(rdf2):
				to_iso.append(claim_ID)
			else:
				match_flag2=1
			if match_flag1 and match_flag2:
				match_fail.append(claim_ID)
			elif match_flag1:
				print(str(claim_ID)+" to_iso failed")
			elif match_flag2:
				print(str(claim_ID)+" iso failed")

	return errorclaimid1,errorclaimid2,iso,to_iso,match_fail,parse_fail,parse1_fail,parse2_fail

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Separate claims into True and False')
	parser.add_argument('-f','--file', metavar='file name',type=str,help='Name of the csv file that stores the claim data',default='claimreviews_db.csv')
	parser.add_argument('-rdf','--rdfpath', metavar='rdf path',type=str,help='RDF path to read the file and store the new files',default='')
	args=parser.parse_args()
	true_claimIDs,false_claimIDs=separate_claims(args.file,args.rdfpath)
	claims_path1='C:\\Users\\zoya\\Desktop\\Zoher\\factcheckgraph\\rdf_files'
	claims_path2='C:\\Users\\zoya\\Desktop\\Zoher\\factcheckgraph\\rdf_files2'
	t1,t2,tiso,tto_iso,tmatch_fail,tparse_fail,tparse1_fail,tparse2_fail=count_claims(claims_path1,claims_path2+"\\true_claims",true_claimIDs)
	f1,f2,fiso,fto_iso,fmatch_fail,fparse_fail,fparse1_fail,fparse2_fail=count_claims(claims_path1,claims_path2+"\\false_claims",false_claimIDs)
	import pdb
	pdb.set_trace()