# from neo4j import GraphDatabase

# uri = "bolt://127.0.0.1:11001"
# driver = GraphDatabase.driver(uri, auth=("neo4j", "1234"))

# def print_rels(tx, name):
# 	for record in tx.run("MATCH (n:fred)-[r{label:{name}}]->(m:dbpedia)"
# 		"where not n.label='Of' and not n.label='Thing'"
# 		"return n.label,r.label,m.label", name=name):
# 		print(record["n.label"],record["r.label"],record["m.label"])

# with driver.session() as session:
# 	session.read_transaction(print_rels,'equivalentClass')
import pandas as pd
import re
import sys
import numpy as np
import rdflib
from rdflib import BNode, Literal
from rdflib.namespace import RDF
from py2neo import Graph, NodeMatcher
data=pd.read_csv("C:/Users/zoya/Desktop/Zoher/factcheckgraph/rdf_files/claimreviews_db.csv",index_col=0)
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
# print(len(falseclaims))
# print(len(data))
# graph = rdflib.Graph()
# truerror=[]
# falserror=[]
# for t in trueclaims:
# 	filename=str(t)+".rdf"
# 	try:
# 		graph.parse(filename,format='application/rdf+xml')
# 	except:
# 		truerror.append(t)
# graph.serialize(destination='truegraph.rdf', format='application/rdf+xml')
# print("True Total:",len(trueclaims))
# print("True Errors:",len(truerror))
# print("True Delta:",len(trueclaims)-len(truerror))

# graph = Graph()

# for f in falseclaims:
# 	filename=str(f)+".rdf"
# 	try:
# 		graph.parse(filename,format='application/rdf+xml')
# 	except:
# 		falserror.append(f)
# graph.serialize(destination='falsegraph.rdf', format='application/rdf+xml')
# print("False Total:",len(falseclaims))
# print("False Errors:",len(falserror))
# print("False Delta:",len(falseclaims)-len(falserror))

# np.save("truerror_claimID.npy",truerror)
# np.save("falserror_claimID.npy",truerror)
# data=np.load("Error500_claimID.npy")
# print(len(data))
# def plotFredGraph(claim_g,filename):
# 	plt.figure()
# 	pos=nx.spring_layout(claim_g)
# 	nx.draw(claim_g,pos,labels={node:node.split("/")[-1].split("#")[-1] for node in claim_g.nodes()},node_size=400)
# 	edge_labels={(edge[0], edge[1]): edge[2]['label'].split("/")[-1].split("#")[-1] for edge in claim_g.edges(data=True)}
# 	nx.draw_networkx_edge_labels(claim_g,pos,edge_labels)
# 	plt.axis('off')
# 	plt.savefig(filename)
# 	plt.close()
# 	plt.clf()
mode=sys.argv[1]
for f in eval(mode+"claims"):
	print(f)
	g=rdflib.Graph()
	# filename="5005"+".rdf"
	filename=str(f)+".rdf"
	try:
		g.parse("C:/Users/zoya/Desktop/Zoher/factcheckgraph/rdf_files/"+mode+"_claims/claim"+filename,format='application/rdf+xml')
	except:
		# continue
		pass
	for subject,predicate,obj in g:
		g.add( (subject, RDF.type,Literal("claim"+filename.strip(".rdf"))) )
		g.add( (obj, RDF.type,Literal("claim"+filename.strip(".rdf"))) )
	g.serialize(destination="C:/Users/zoya/Desktop/Zoher/factcheckgraph/rdf_files/"+mode+"_claims/"+filename, format='application/rdf+xml')
	graph = Graph("bolt://127.0.0.1:7687",password="1234")
	tx = graph.begin()
	# tx.run("MATCH (n) DETACH DELETE n;")
	print([record for record in tx.run("CALL semantics.importRDF('file:///C:/Users/zoya/Desktop/Zoher/factcheckgraph/rdf_files/"+mode+"_claims/"+filename+"','RDF/XML', { shortenUrls: false, typesToLabels: false, commitSize: 100000 });")])
	tx.run("MATCH (n) where exists(n.`http://www.w3.org/1999/02/22-rdf-syntax-ns#type`) set n.claim{}=True;".format(str(f)))
	tx.run("MATCH (n) where exists(n.`http://www.w3.org/1999/02/22-rdf-syntax-ns#type`) remove n.`http://www.w3.org/1999/02/22-rdf-syntax-ns#type`;")
	#1 Add prefix labels for colors
	tx.run("MATCH (n) where n.uri starts with 'http://www.w3.org/2000/01/rdf-schema#' set n:rdfs;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.w3.org/1999/02/22-rdf-syntax-ns#' set n:rdf;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#' set n:dul;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/vn/abox/role/' set n:vnrole;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/vn/' set n:vn;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#' set n:boxing;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/boxer/boxer.owl#' set n:boxer;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/d0.owl#' set n:d0;")
	tx.run("MATCH (n) where n.uri starts with 'http://schema.org/' set n:schemaorg;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.essepuntato.it/2008/12/earmark#' set n:earmark;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/fred/domain.owl#' set n:fred;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/fred/quantifiers.owl#' set n:fredquant;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.w3.org/2006/03/wn/wn30/schema/' set n:wn;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/fred/pos.owl#' set n:pos;")
	tx.run("MATCH (n) where n.uri starts with 'http://ontologydesignpatterns.org/cp/owl/semiotics.owl#' set n:semiotics;")
	tx.run("MATCH (n) where n.uri starts with 'http://www.w3.org/2002/07/owl#' set n:owl;")
	tx.run("MATCH (n) where n.uri starts with 'http://dbpedia.org/resource/' set n:dbpedia;")
	# #2 labeling the nodes
	tx.run("MATCH (n) with n, SPLIT(n.uri,'/')[-1] as name SET n.label_name=name;")
	tx.run("MATCH (n) with n, SPLIT(n.label_name,'#')[-1] as name SET n.label_name=name;")
	# #3 labeling the relationships
	tx.run("MATCH ()-[r]-() with r, SPLIT(type(r),'/')[-1] as name SET r.label_name=name;")
	tx.run("MATCH ()-[r]-() with r, SPLIT(r.label_name,'#')[-1] as name SET r.label_name=name;")
	#4 Delete '%27'
	tx.run("MATCH (n) with n,split(n.label_name,'_') as splitn where n.label_name='%27' or (splitn[0]=~'%27' and splitn[1]=~'[0-9]+') DETACH delete n;")
	#6 merging verbs like show_1 and show_2 with show
	tx.run("MATCH (n:fred)-[r:`http://www.w3.org/1999/02/22-rdf-syntax-ns#type`]->(m) with n,r,m,split(n.label_name,'_') as splitn where splitn[0]=toLower(m.label_name) and splitn[1]=~'[0-9]+' with collect([n,m]) as events unwind events as event CALL apoc.refactor.mergeNodes([event[0],event[1]],{properties:'overwrite',mergeRels:true}) yield node return node;")
	#delete self loops
	tx.run("MATCH (n)-[r]-(n) delete r;")
	tx.run("MATCH (n:fred)-[r:`http://www.w3.org/1999/02/22-rdf-syntax-ns#type`]->(m) with n,r,m,split(n.label_name,'_') as splitn where n.label_name contains '_' and splitn[1]=~'[0-9]+' and toLower(m.label_name) contains splitn[0] with collect([n,m]) as events unwind events as event CALL apoc.refactor.mergeNodes([event[0],event[1]],{properties:'overwrite',mergeRels:true}) yield node return node;")
	tx.run("MATCH (n)-[r]-(n) delete r;")
	#remove merged label_names
	#Merging verbs with their verbnet forms
	tx.run("MATCH (n:fred)-[r:`http://www.w3.org/2002/07/owl#equivalentClass`]->(m:vn) where toLower(split(n.label_name,'_')[0])=toLower(split(m.label_name,'_')[0]) with collect([n,m]) as events unwind events as event CALL apoc.refactor.mergeNodes([event[0],event[1]],{properties:'overwrite',mergeRels:true}) yield node return node;")
	tx.run("MATCH (n)-[r]-(n) delete r;")
	tx.run("MATCH (n:vn) remove n:fred;")
	#4 merging nodes with sameAs relationships
	tx.run("MATCH (n:fred)-[r:`http://www.w3.org/2002/07/owl#sameAs`]->(m:dbpedia) where not n.label_name='Of' and not n.label_name='Thing' with collect([n,m]) as events unwind events as event CALL apoc.refactor.mergeNodes([event[0],event[1]],{properties:'overwrite',mergeRels:true}) yield node return node;")
	#delete self loops
	tx.run("MATCH (n)-[r]-(n) delete r;")
	#5 merging nodes with equivalentClass relationships
	tx.run("MATCH (n:fred)-[r:`http://www.w3.org/2002/07/owl#equivalentClass`]->(m:dbpedia) where not n.label_name='Of' and not n.label_name='Thing' with collect([n,m]) as events unwind events as event CALL apoc.refactor.mergeNodes([event[0],event[1]],{properties:'overwrite',mergeRels:true}) yield node return node;")
	#delete self loops
	tx.run("MATCH (n)-[r]-(n) delete r;")
	#5 merging nodes with associatedWith relationships
	tx.run("MATCH (n:fred)-[r:`http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#associatedWith`]-(m:dbpedia) where not n.label_name='Of' and not n.label_name='Thing' and (toLower(split(n.label_name,'_')[0])=toLower(split(m.label_name,'_')[0])) with collect([n,m]) as events unwind events as event CALL apoc.refactor.mergeNodes([event[0],event[1]],{properties:'overwrite',mergeRels:true}) yield node return node;")
	#delete self loops
	#remove merged label_names
	tx.run("MATCH (n:dbpedia) remove n:fred;")
	tx.run("MATCH (n)-[r]-(n) delete r;")
	#remove hasDeterminer 'The'
	tx.run("MATCH (n)-[r:`http://www.ontologydesignpatterns.org/ont/fred/quantifiers.owl#hasDeterminer`]->(m) delete m,r")
	#
	tx.run("MATCH (n{uri:'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasDataValue'}) detach delete n")
	#
	tx.run("MATCH (n{uri:'http://www.w3.org/2002/07/owl#DataTypeProperty'}) detach delete n")
	#
	tx.commit()
	print(tx.finished())
tx = graph.begin()
tx.run("MATCH (n)-[r]-(n) delete r;")
# tx.run("MATCH (n) set n:"+"claim"+filename.strip(".rdf"))
tx.run("Match(n) where not n.uri starts with 'http://' detach delete (n)")
tx.commit()
print(tx.finished())


# g=rdflib.Graph()
# filename="1"+".rdf"
# # filename=str(f)+".rdf"
# try:
# 	g.parse("C:/Users/zoya/Desktop/Zoher/factcheckgraph/RDF Files/"+filename,format='application/rdf+xml')
# except:
# 	# continue
# 	pass
# for subject,predicate,obj in g:
# 	g.add( (subject, RDF.type,Literal("claim"+filename.strip(".rdf"))) )
# 	g.add( (obj, RDF.type,Literal("claim"+filename.strip(".rdf"))) )
# g.serialize(destination="claim"+filename, format='application/rdf+xml')
# filename="claim"+filename
# graph = Graph("bolt://127.0.0.1:7687",password="1234")
# tx = graph.begin()
# # tx.run("MATCH (n) DETACH DELETE n;")
# print([record for record in tx.run("CALL semantics.importRDF('file:///C:/Users/zoya/Desktop/Zoher/factcheckgraph/RDF Files/"+filename+"','RDF/XML', { shortenUrls: false, typesToLabels: false, commitSize: 100000 });")])
# tx.run("MATCH (n) where exists(n.`http://www.w3.org/1999/02/22-rdf-syntax-ns#type`) set n."+filename.strip(".rdf")+"=True")
# tx.run("MATCH (n) where exists(n.`http://www.w3.org/1999/02/22-rdf-syntax-ns#type`) remove n.`http://www.w3.org/1999/02/22-rdf-syntax-ns#type`;")
# #1 Add prefix labels for colors
# tx.run("MATCH (n) where n.uri starts with 'http://www.w3.org/2000/01/rdf-schema#' set n:rdfs;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.w3.org/1999/02/22-rdf-syntax-ns#' set n:rdf;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#' set n:dul;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/vn/abox/role/' set n:vnrole;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/vn/' set n:vn;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/boxer/boxing.owl#' set n:boxing;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/boxer/boxer.owl#' set n:boxer;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/d0.owl#' set n:d0;")
# tx.run("MATCH (n) where n.uri starts with 'http://schema.org/' set n:schemaorg;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.essepuntato.it/2008/12/earmark#' set n:earmark;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/fred/domain.owl#' set n:fred;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/fred/quantifiers.owl#' set n:fredquant;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.w3.org/2006/03/wn/wn30/schema/' set n:wn;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.ontologydesignpatterns.org/ont/fred/pos.owl#' set n:pos;")
# tx.run("MATCH (n) where n.uri starts with 'http://ontologydesignpatterns.org/cp/owl/semiotics.owl#' set n:semiotics;")
# tx.run("MATCH (n) where n.uri starts with 'http://www.w3.org/2002/07/owl#' set n:owl;")
# tx.run("MATCH (n) where n.uri starts with 'http://dbpedia.org/resource/' set n:dbpedia;")
# tx.run("MATCH (n) where n.uri starts with 'http://dbpedia.org/resource/' set n:dbpedia;")
# tx.run("MATCH (n) where n.uri starts with 'http://dbpedia.org/resource/' set n:dbpedia;")
# #2 labeling the nodes
# tx.run("MATCH (n) with n, SPLIT(n.uri,'/')[-1] as name SET n.label_name=name;")
# tx.run("MATCH (n) with n, SPLIT(n.label_name,'#')[-1] as name SET n.label_name=name;")
# #3 labeling the relationships
# tx.run("MATCH ()-[r]-() with r, SPLIT(type(r),'/')[-1] as name SET r.label_name=name;")
# tx.run("MATCH ()-[r]-() with r, SPLIT(r.label_name,'#')[-1] as name SET r.label_name=name;")
# #4 Delete '%27'
# tx.run("MATCH (n) with n,split(n.label_name,'_') as splitn where n.label_name='%27' or (splitn[0]=~'%27' and splitn[1]=~'[0-9]+') DETACH delete n;")
# #6 merging verbs like show_1 and show_2 with show
# tx.run("MATCH (n:fred)-[r:`http://www.w3.org/1999/02/22-rdf-syntax-ns#type`]->(m) with n,r,m,split(n.label_name,'_') as splitn where splitn[0]=toLower(m.label_name) and splitn[1]=~'[0-9]+' with collect([n,m]) as events unwind events as event CALL apoc.refactor.mergeNodes([event[0],event[1]],{properties:'overwrite',mergeRels:true}) yield node return node;")
# #delete self loops
# tx.run("MATCH (n)-[r]-(n) delete r;")
# tx.run("MATCH (n:fred)-[r:`http://www.w3.org/1999/02/22-rdf-syntax-ns#type`]->(m) with n,r,m,split(n.label_name,'_') as splitn where n.label_name contains '_' and splitn[1]=~'[0-9]+' and toLower(m.label_name) contains splitn[0] with collect([n,m]) as events unwind events as event CALL apoc.refactor.mergeNodes([event[0],event[1]],{properties:'overwrite',mergeRels:true}) yield node return node;")
# tx.run("MATCH (n)-[r]-(n) delete r;")
# #remove merged label_names
# #Merging verbs with their verbnet forms
# tx.run("MATCH (n:fred)-[r:`http://www.w3.org/2002/07/owl#equivalentClass`]->(m:vn) where toLower(split(n.label_name,'_')[0])=toLower(split(m.label_name,'_')[0]) with collect([n,m]) as events unwind events as event CALL apoc.refactor.mergeNodes([event[0],event[1]],{properties:'overwrite',mergeRels:true}) yield node return node;")
# tx.run("MATCH (n)-[r]-(n) delete r;")
# tx.run("MATCH (n:vn) remove n:fred;")

# #4 merging nodes with sameAs relationships
# tx.run("MATCH (n:fred)-[r:`http://www.w3.org/2002/07/owl#sameAs`]->(m:dbpedia) where not n.label_name='Of' and not n.label_name='Thing' with collect([n,m]) as events unwind events as event CALL apoc.refactor.mergeNodes([event[0],event[1]],{properties:'overwrite',mergeRels:true}) yield node return node;")
# #delete self loops
# tx.run("MATCH (n)-[r]-(n) delete r;")
# #5 merging nodes with equivalentClass relationships
# tx.run("MATCH (n:fred)-[r:`http://www.w3.org/2002/07/owl#equivalentClass`]->(m:dbpedia) where not n.label_name='Of' and not n.label_name='Thing' with collect([n,m]) as events unwind events as event CALL apoc.refactor.mergeNodes([event[0],event[1]],{properties:'overwrite',mergeRels:true}) yield node return node;")
# #delete self loops
# tx.run("MATCH (n)-[r]-(n) delete r;")
# #5 merging nodes with associatedWith relationships
# tx.run("MATCH (n:fred)-[r:`http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#associatedWith`]-(m:dbpedia) where not n.label_name='Of' and not n.label_name='Thing' and (toLower(split(n.label_name,'_')[0])=toLower(split(m.label_name,'_')[0])) with collect([n,m]) as events unwind events as event CALL apoc.refactor.mergeNodes([event[0],event[1]],{properties:'overwrite',mergeRels:true}) yield node return node;")
# #delete self loops
# tx.run("MATCH (n)-[r]-(n) delete r;")
# #remove merged label_names
# tx.run("MATCH (n:dbpedia) remove n:fred;")
# #remove hasDeterminer 'The'
# tx.run("MATCH (n)-[r:`http://www.ontologydesignpatterns.org/ont/fred/quantifiers.owl#hasDeterminer`]->(m) delete m,r")
# #
# tx.run("MATCH (n{uri:'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasDataValue'}) detach delete n")
# #
# tx.run("MATCH (n{uri:'http://www.w3.org/2002/07/owl#DataTypeProperty'}) detach delete n")
# #
# # tx.run("MATCH (n) set n:"+"claim"+filename.strip(".rdf"))
# tx.commit()
# print(tx.finished())