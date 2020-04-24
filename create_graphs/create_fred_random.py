rdf_path="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/covid19_rdf_files/"
claim_type="covid19"
claims_path=os.path.join(rdf_path,"{}_claims".format(claim_type))
claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
init=0
end=len(claims)
index=0
graph_path="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/graphs/covid19"
compilefred=1
cpu=1
claims=pd.read_csv(os.path.join(rdf_path,"{}_claims.csv".format(claim_type)))
claim_IDs=claims['claimID'].tolist()
index,errorclaimid,clean_claims=passiveFredParse(index,claims_path,claim_IDs[:20],init,end)
# errorclaimid,clean_claims=fredParse(claims_path,claims,init,end)
np.save(os.path.join(rdf_path,"{}_error_claimID.npy".format(claim_type)),errorclaimid)
with codecs.open(os.path.join(rdf_path,"{}claims_clean.json".format(claim_type)),"w","utf-8") as f:
	f.write(json.dumps(clean_claims,indent=4,ensure_ascii=False))

if compilefred!=0:
	with codecs.open(os.path.join(rdf_path,"{}claims_clean.json".format(claim_type)),"r","utf-8") as f: 
		clean_claims=json.loads(f.read())
	if compilefred==1 or compilefred==2:
		# n=int(len(claims)/cpu)+1
		# pool=mp.Pool(processes=cpu)							
		# results=[pool.apply_async(eval("compileClaimGraph"+str(compilefred)), args=(index,claims_path,claim_IDs,clean_claims,index*n,(index+1)*n)) for index in range(cpu)]
		# output=sorted([p.get() for p in results],key=lambda x:x[0])
		# fcgs=list(map(lambda x:x[1],output))
		# master_fcg=nx.Graph()
		# for fcg in fcgs:
		# 	master_fcg=nx.compose(master_fcg,fcg)
		master_fcg=list(compileClaimGraph1(0,claims_path,claim_IDs,clean_claims,0,len(claims)+1))[1]
		if compilefred==2:
			master_clean=compile_clean(rdf_path,clean_claims,claim_type)
			master_fcg=cleanClaimGraph(master_fcg,master_clean)
	elif compilefred==3:
		master_fcg=compileClaimGraph3(claims_path,claim_IDs,clean_claims)
	saveFred(master_fcg,graph_path,claim_type+str(compilefred),compilefred)