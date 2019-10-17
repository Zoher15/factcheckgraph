def find_intersect(graph_path,fcg_class,kg_label):
	fcg_labels={"fred":{"true":"TFCG","false":"FFCG"},"co-occur":{"true":"TFCG_co","false":"FFCG_co"}}
	tfcg_entities=np.load(os.path.join(graph_path,fcg_class,"{}_entities.npy".format(fcg_labels[fcg_class]["true"])))
	ffcg_entities=np.load(os.path.join(graph_path,fcg_class,"{}_entities.npy".format(fcg_labels[fcg_class]["false"])))
	kg_entities=np.load(os.path.join(graph_path,fcg_class,"{}_entities.npy".format(kg_label)))
	intersect_entities=np.asarray(list(set(tfcg_entities).intersection(set(ffcg_entities))))
	intersect_entities=np.asarray(list(set(intersect_entities).intersection(set(kg_entities))))
	np.save(os.path.join(graph_path,fcg_class,"Intersect_entities_{}_{}.npy".format(kg_label,fcg_class)),intersect_entities)
	#Finding all possible combinations
	intersect_all_entityPairs=combinations(intersect_entities,2)
	#Converting tuples to list
	intersect_all_entityPairs=np.asarray(list(map(list,intersect_all_entityPairs)))
	np.save(os.path.join(graph_path,fcg_class,"Intersect_all_entityPairs_{}_{}.npy".format(kg_label,fcg_class)),intersect_all_entityPairs)

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Find intersection of entities for graphs')
	parser.add_argument('-gp','--graphpath', metavar='graph path',type=str,help='Path to the graph directory')
	parser.add_argument('-fcg','--fcgclass', metavar='fcg class',type=str,choices=['fred','co-occur','backbone'],help='Class of FactCheckGraph to process')
	parser.add_argument('-kg','--kg', metavar='knowledgegraph type',type=str,choices=['dbpedia','wikidata'],help='DBPedia or Wikidata Graph')
	args=parser.parse_args()
	find_intersect(args.graphpath,args.fcgclass,args.kg)
