import pandas as pd 
import numpy as np
from newspaper import Article
import argparse
import re
import os

def parse_titles(row):
	url=row['review_url']
	try:
		article = Article(url)
		article.download()
		article.parse()
		article.nlp()
		row['title_text']=article.title
		row['article_text']=article.text
		row["keywords"]=article.keywords
		row["summary"]=article.summary
	except:
		print("Exception:",row['claimID'])
	return row

def extract_titles(file,rdf_path):
	data=pd.read_csv(os.path.join(rdf_path,file),index_col=0)
	##Dropping non-str rows
	filter=list(map(lambda x:type(x)!=str,data['rating_name']))
	data.drop(data[filter].index,inplace=True)
	#drop duplicates with the same claimID
	data.drop_duplicates('claimID',keep='first',inplace=True)
	data["title_text"]=""
	data["article_text"]=""
	data["keywords"]=""
	data["summary"]=""
	data=data.apply(parse_titles,axis=1)
	data.to_csv(os.path.join(rdf_path,file))
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
	parser=argparse.ArgumentParser(description='Extract titles from claims')
	parser.add_argument('-f','--file', metavar='file name',type=str,help='Name of the csv file that stores the claim data',default='claimreviews_db.csv')
	parser.add_argument('-rdf','--rdfpath', metavar='rdf path',type=str,help='RDF path to read the file and store the new files',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/rdf_files')
	args=parser.parse_args()
	extract_titles(args.file,args.rdfpath)