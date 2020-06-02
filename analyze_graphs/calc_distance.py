from sentence_transformers import SentenceTransformer
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

def calc_distance(sent1,sent2,model_path):
	model = SentenceTransformer(model_path)
	sent1=model.encode([sent1])
	sent2=model.encode([sent2])
	simil_p=cosine_similarity(sent1,sent2)[0]
	simil_p=1-np.arccos(simil_p)/np.pi
	print(simil_p)

if __name__== "__main__":
	parser=argparse.ArgumentParser(description='Find angular distance between sentences')
	parser.add_argument('-s1','--sent1', metavar='sentence 1',type=str)
	parser.add_argument('-s2','--sent2', metavar='sentence 2',type=str)
	parser.add_argument('-mp','--modelpath', metavar='model path',type=str,help='Model directory to load the model',default="/geode2/home/u110/zkachwal/BigRed3/factcheckgraph_data/models/claims-relatedness-model/claims-roberta-base-nli-stsb-mean-tokens-2020-05-27_19-01-27")
	args=parser.parse_args()
	calc_distance(args.sent1,args.sent2,args.modelpath)