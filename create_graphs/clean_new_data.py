import os
import pandas as pd
import multiprocessing as mp
import pandas as pd 
from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect
import numpy as np
import argparse
import pdb
import re
import os

'''
{'aap.com.au':['australia'], 'abc.net.au':['australia'], 'africacheck.org':['south africa','kenya','nigeria','senegal'],
       'altnews.in':['india'],'ballotpedia.org':['usa'], 'bbc.co.uk':['uk'], 'bbc.com':['global'],
       'boomlive.in':['india'],'caravanmagazine.in':['india'], 'cbsnews.com':['usa'],
       'channel4.com:['britain']','climatefeedback.org':['global'], 'factcheck.aap.com.au':['australia'],
       'factcheck.afp.com':['france'],'factcheck.org':['usa'],
       'factcheck.thedispatch.com':['usa'], 'factchecker.in':['india'], 'factcheckni.org':['norther ireland'],
       'factcheckthailand.afp.com':['thailand'],'factly.in':['india'],
       'factrakers.org:['phillipines']', 'factscan.ca':['canada'],'francetvinfo.fr', 'fullfact.org', 'guardian.ng',
       'gujarati.factcrescendo.com', 'healthfeedback.org',
       'hindi.asianetnews.com', 'hindi.boomlive.in', 'icirnigeria.org',
       'indiatoday.in', 'kannada.asianetnews.com',
       'legendsoflocalization.com', 'liberation.fr',
       'malayalam.factcrescendo.com', 'malayalam.samayam.com',
       'maldita.es', 'marathi.factcrescendo.com', 'mskcc.org',
       'namibiafactcheck.org.na', 'newsable.asianetnews.com',
       'newsweek.com', 'newswise.com', 'newtral.es',
       'noticias.uol.com.br', 'nytimes.com', 'pagellapolitica.it',
       'periksafakta.afp.com', 'piaui.folha.uol.com.br',
       'poligrafo.sapo.pt', 'politica.estadao.com.br', 'politifact.com',
       'polygraph.info', 'precisionvaccinations.com', 'rappler.com',
       'satyagrah.scroll.in', 'sciencefeedback.co', 'scroll.in',
       'semakanfakta.afp.com', 'sites.nationalacademies.org',
       'snopes.com', 'sprawdzam.afp.com', 'tamil.factcrescendo.com',
       'teyit.org', 'tfc-taiwan.org.tw', 'theconversation.com',
       'theferret.scot', 'thegazette.com', 'theglobeandmail.com',
       'thelogicalindian.com', 'thequint.com',
       'timesofindia.indiatimes.com', 'tsek.ph', 'verafiles.org',
       'vox.com', 'washingtonexaminer.com', 'washingtonpost.com',
       'weeklystandard.com', 'youturn.in', 'zimfact.org'}
'''

def lang_detect(text):
	try:
		ret=detect(text)
	except:
		ret=None
	return ret

def clean_new_data(file,rdf_path):
	data=pd.read_csv(os.path.join(rdf_path,file))
	##Dropping non-str rows
	filter=list(map(lambda x:type(x)!=str,data['claim_text']))
	data.drop(data[filter].index,inplace=True)
	filter=list(map(lambda x:type(x)!=str,data['verdict']))
	data.drop(data[filter].index,inplace=True)
	data['claim_text']=data['claim_text'].apply(lambda x:eval(x)[0])
	data['verdict']=data['verdict'].apply(lambda x:eval(x)[0])
	data['lang']=data['claim_text'].apply(lambda x:lang_detect(x))
	data=data[data['lang']=='en']
	print(data.groupby('publisher').count())
	# pdb.set_trace()
	# data['vlang']=data['verdict'].apply(lambda x:lang_detect(x))
	# data=data[data['vlang']=='en']
	data.to_csv(os.path.join(rdf_path,file.split(".csv")[0]+"_en.csv"))

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Clean new data')
	parser.add_argument('-f','--file', metavar='file name',type=str,help='Name of the csv file that stores the claim data',default='fcc.csv')
	parser.add_argument('-r','--rdfpath', metavar='rdf path',type=str,help='Path to the rdf files parsed by FRED',default='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/rdf_files/')
	args=parser.parse_args()
	clean_new_data(args.file,args.rdfpath)