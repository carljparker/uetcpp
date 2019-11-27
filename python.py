import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np 
from numpy import genfromtxt
# from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as NMI
from sklearn.datasets import load_iris
import random
import math
from joblib import Parallel, delayed
import multiprocessing
from sklearn import preprocessing
import time
import uetlib
from sklearn import datasets


def sim_to_dist(sim):
    out = np.zeros((len(sim),len(sim)))
    for i in range(len((sim))):
        for j in range(len(sim)):
            out[i][j] = (1 - sim[i][j])
    return(out)

def build_ensemble_inc(data,n_estimators=250,nmin=None,coltypes=None):
	if nmin == None:
		nmin = math.floor(len(data)/3)
	similarities = np.zeros((len(data),len(data)))
	num_cores = multiprocessing.cpu_count()
	results = Parallel(n_jobs=num_cores)(delayed(uetlib.get_sim_one)(data,nmin,coltypes) for i in range(n_estimators))
	similarities = results
	leafSize = []
	# for a,b,c in results:
	#     leafSize.extend(b)
	# np.savetxt("sizes.txt", leafSize, delimiter = ",", fmt='%.d')
	# print(np.mean([a[2] for a in results]))
	return(np.sum(similarities,axis=0)/n_estimators)

def test_clustering(data, Y, coltype):
    nmis = []
    times = []
    n_clusters = len(set(Y))
    for i in range(5):
        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed",linkage="average")
        start = time.time()
        sim = build_ensemble_inc(data, coltypes=coltype)
        duration = time.time() - start
        times.append(duration)
        distance = sim_to_dist(sim)
        # print(distance)
        predicted = cluster.fit_predict(distance)
        nmis.append(NMI(Y,predicted))
    print("Summary of all the runs.\n")
    print("Mean nmi : {0} (standard deviation : {1}), mean duration : {2} (standard deviation : {3}) \n".format(np.mean(nmis),np.std(nmis),np.mean(duration), np.std(duration)))
    print("----------\n")


data_iris, Y = load_iris(return_X_y=True)
iris_type = [0 for i in range(4)]

n_clusters = len(set(Y))

# test_clustering(data_iris, Y, iris_type)
df = pd.read_csv("./matrix.csv", sep="\t", header=None)   # read dummy .tsv file into memory

sim = df.values  # access the numpy array containing values

# sim = build_ensemble_inc(data_iris, iris_type)
cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed",linkage="average")
distance = sim_to_dist(sim)

predicted = cluster.fit_predict(distance)
print(NMI(predicted, Y))

# data_iris = datasets.load_iris()
# iris_type = [0 for i in range(4)]
# sim = build_ensemble_inc(data_iris.data, coltypes=iris_type)
# distance2 = sim_to_dist(sim)
# np.savetxt("./matrix2.csv", distance2, delimiter="\t", fmt="%1.4f")
# distance2 = pd.read_csv("./matrix2.csv", delimiter="\t", header=None)
# predicted = cluster.fit_predict(distance2.values)
# print(NMI(predicted, Y))
