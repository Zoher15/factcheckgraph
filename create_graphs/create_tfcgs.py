import os
import pandas as pd
import multiprocessing as mp
import numpy as np
from create_co_occur import *

rdf_path='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/rdf_files/'
graph_path='/gpfs/home/z/k/zkachwal/BigRed3/factcheckgraph_data/graphs/'
fcg_label='tfcg_co'
claimIDs=list(np.load(os.path.join(rdf_path,"{}_claimID.npy".format('true'))))
pool=mp.Pool(processes=48)	
results=[pool.apply_async(create_co_occur, args=(rdf_path,graph_path,fcg_label,ID)) for ID in claimIDs]
output=[p.get() for p in results]