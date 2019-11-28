import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomTreesEmbedding as rte
from sklearn.cluster.hierarchical import AgglomerativeClustering as hac
import math
import warnings
import random
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import markov_clustering as mc
import community as louvain
import os
import json
from networkx.generators import community, erdos_renyi_graph
from sklearn import preprocessing
from sklearn import datasets

plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)


import subprocess

def ensemble_attributes(path, sep, coltypes):
    cmd = ['./main', '-p', "{0}".format(path), "-c", coltypes]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p.wait()
    return(p.returncode)

def ensemble_attributes_dense(path, sep, coltypes):
    cmd = ['./main', '-p', "{0}".format(path), "-c", coltypes, "-m", "1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p.wait()
    return(p.returncode)

def test_clustering(n_runs=5, alpha=0.5, coltypes=None):
    nmis_attributes = []
    colTypesString = ""
    for i in coltypes:
        colTypesString+="{0}".format(i)
    for i in range(n_runs):
        print("Run number {0}".format(i))
        ensemble_attributes("file_attributes.csv", "\t", coltypes=colTypesString)
        sim_attributes = pd.read_csv("./matrix_uet.csv", delimiter="\t", header=None).values
        sim_attributes = sim_attributes[:,:-1]

        dist_attributes = sim_to_dist(np.array(sim_attributes))
        model = hac(n_clusters=len(set(true)), affinity="precomputed", linkage="average")
        # model_kmeans = KMeans(n_clusters=len(set(true)))
        # scaler = QuantileTransformer(n_quantiles=10)

        # dist_attributes_scaled = scaler.fit_transform(dist_attributes)

        # results_dense_attributes = TSNE(metric="precomputed").fit_transform(dist_attributes_scaled)
        labels_attributes = model.fit_predict(dist_attributes)

        nmis_attributes.append(nmi(labels_attributes, true, average_method="arithmetic"))
    print("Attributes : {0}, {1}".format(np.mean(nmis_attributes), np.std(nmis_attributes)))

    return(nmis_attributes)

def test_clusteringDense(n_runs=5, alpha=0.5, coltypes=None):
    nmis_attributes = []
    colTypesString = ""
    for i in coltypes:
        colTypesString+="{0}".format(i)
    for i in range(n_runs):
        print("Run number {0}".format(i))

        ensemble_attributes_dense("file_attributes.csv", "\t", coltypes=colTypesString)
        sim_attributes = pd.read_csv("./matrix_uet.csv", delimiter="\t", header=None).values
        sim_attributes = sim_attributes[:,:-1]

        dist_attributes = np.array(sim_attributes)
        model = hac(n_clusters=len(set(true)), affinity="precomputed", linkage="average")
        # model_kmeans = KMeans(n_clusters=len(set(true)))
        # scaler = QuantileTransformer(n_quantiles=10)

        # dist_attributes_scaled = scaler.fit_transform(dist_attributes)

        # results_dense_attributes = TSNE(metric="precomputed").fit_transform(dist_attributes_scaled)
        labels_dense_attributes = model.fit_predict(dist_attributes)

        nmis_attributes.append(nmi(labels_dense_attributes, true, average_method="arithmetic"))
    print("Attributes : {0}, {1}".format(np.mean(nmis_attributes), np.std(nmis_attributes)))

    return(nmis_attributes)

colors = []
r = lambda: random.randint(0,255)

for i in range(1000):
    colors.append('#%02X%02X%02X' % (r(),r(),r()))

def sim_to_dist(matrix):
    out = np.zeros((len(matrix),len(matrix)))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            out[i][j] = math.sqrt(1 - matrix[i][j])
    return(out)

print("----------")
print("Synthetic datasets")
print("----------")

a = np.random.uniform(0,0.5,250)
a = np.append(a,np.random.uniform(0.5,1,250))
b = np.random.uniform(1,2,250)
b = np.append(b,np.random.uniform(0,1,250))
data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b)))
X = data_cluster_corr.values.astype('float64')
np.savetxt('file_attributes.csv', X, delimiter='\t')

true = [0 if i < 250 else 1 for i in range(len(data_cluster_corr))]
coltypes = [0,0]
print("Test dataset (with clusters - Continuous) - 500 samples. 2 continuous columns, 2 classes")
test_clustering(coltypes=coltypes)
test_clusteringDense(coltypes=coltypes)


a = np.random.uniform(0,0.5,250)
a = np.append(a,np.random.uniform(0.5,1,250))
b = np.random.uniform(1,2,250)
b = np.append(b,np.random.uniform(0,1,250))

c = np.random.uniform(0,0.2,250)
c = np.append(c,np.random.uniform(0.4,0.6,250))
d = np.random.uniform(4,5,250)
d = np.append(d, np.random.uniform(0,1,250))
data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b,c,d)))

X = data_cluster_corr.values.astype('float64')
np.savetxt('file_attributes.csv', X, delimiter='\t')

true = [0 if i < 250 else 1 for i in range(len(data_cluster_corr))]
coltypes = [0,0,0,0]
print("Test dataset (with clusters - Continuous) - 500 samples. 4 continuous columns, 2 classes")
test_clustering(coltypes=coltypes)
test_clusteringDense(coltypes=coltypes)


print("----------")
print("Real datasets")
print("----------")

print("----------")
print("Iris dataset")
print("----------")

data_iris = datasets.load_iris()
X = np.array(data_iris.data)
np.savetxt('file_attributes.csv', X, delimiter='\t')

true = np.array(data_iris.target)
coltypes = [0 for i in range(4)]
print("Test Iris")
test_clustering(coltypes=coltypes)
test_clusteringDense(coltypes=coltypes)

print("----------")
print("Wisconsin dataset")
print("----------")

data_wisc = pd.read_csv("./data/wisconsin.txt",sep="\t",header=None).astype('float64')
coltypes = [0 for i in range(len(data_wisc.columns))]
X = data_wisc.iloc[:,:-1].values
np.savetxt('file_attributes.csv', X, delimiter='\t')

true = data_wisc.iloc[:,-1].values
print("Wisconsin dataset")
test_clustering(coltypes=coltypes)
test_clusteringDense(coltypes=coltypes)

print("----------")
print("Madelon dataset")
print("----------")

data_madelon = pd.read_csv("./data/madelon.txt",sep="\t",header=None).astype('float64')
madelon_type = [0 for i in range(len(data_madelon.columns))]
X = data_madelon.iloc[:,:-1].values
np.savetxt('file_attributes.csv', X, delimiter='\t')
true = data_madelon.iloc[:,-1].values
coltypes = madelon_type
print("Madelon dataset")
test_clustering(coltypes=coltypes)
test_clusteringDense(coltypes=coltypes)

print("----------")
print("Isolet dataset")
print("----------")

data_isolet  = pd.read_csv('./data/isolet.txt',sep="\t",header=None).astype('float64')
isolet_type = [0 for i in range(len(data_isolet.columns))]
X = data_isolet.iloc[:,:-1].values
true = data_isolet.iloc[:,-1]
coltypes = isolet_type
np.savetxt('file_attributes.csv', X, delimiter='\t')

print("Isolet dataset")
test_clustering(coltypes=coltypes)
test_clusteringDense(coltypes=coltypes)

print("----------")
print("Pima dataset")
print("----------")

data_pima = pd.read_csv('./data/Pima.txt',sep="\t",header=None).astype('float64')
pima_type = [0 for i in range(len(data_pima.columns))]
X = data_pima.iloc[:,:-1].values
true = data_pima.iloc[:,-1].values
coltypes = pima_type
np.savetxt('file_attributes.csv', X, delimiter='\t')

print("Pima dataset")
test_clustering(coltypes=coltypes)
test_clusteringDense(coltypes=coltypes)

print("----------")
print("Spam base dataset")
print("----------")

data_spam = pd.read_csv("./data/spamb.txt",sep="\t",header=None).astype('float64')
spam_type = [0 for i in range(len(data_spam.columns))]
X = data_spam.iloc[:,:-1].values
true = data_spam.iloc[:,-1].values
coltypes = spam_type
np.savetxt('file_attributes.csv', X, delimiter='\t')

print("Spam base dataset")
test_clustering(coltypes=coltypes)
test_clusteringDense(coltypes=coltypes)


print("----------")
print("Soybean dataset")
print("----------")

data_soybean = pd.read_csv("./data/soybean.data",sep="\t",header=None).astype('float64')
soybean_type = [1 for i in range(len(data_soybean.columns))]
X = data_soybean.iloc[:,:-1].values
true =  data_soybean.iloc[:,-1].values
np.savetxt('file_attributes.csv', X, delimiter='\t')
coltypes = soybean_type
print("Soybean dataset")
test_clustering(coltypes=coltypes)
test_clusteringDense(coltypes=coltypes)

# data_credit = pd.read_csv("./data/credit.csv",sep=",")
# data_credit[['c1','c4','c5','c6','c7','c9','c10','c12','c13','c16']] = data_credit[['c1','c4','c5','c6','c7','c9','c10','c12','c13','c16']].apply(preprocessing.LabelEncoder().fit_transform).astype('float64')
# data_credit.replace('?',np.nan, inplace=True)
# data_credit.dropna(axis=0,how='any', inplace=True)
# X = data_credit.astype('float64')
# np.savetxt('file_attributes.csv', X, delimiter='\t')
# credit_type = [1,0,0,1,1,1,1,0,1,1,0,1,1,0,0]
# X = data_credit.iloc[:,:-1].values
# true = data_credit.ix[:,-1].values
# coltypes = credit_type
# print("Credit dataset")
# test_clustering(20, coltypes=coltypes)

# data_cmc = pd.read_csv("./data/cmc.csv",sep=",",header=None).astype('float64')
# cmc_type = [0,1,1,0,1,1,1,1,1]
# X = data_cmc.iloc[:,:-1].values
# true = data_cmc.iloc[:,-1].values
# coltypes = cmc_type
# print("CMC dataset - Num/Cat")
# test_clustering(20, coltypes=coltypes)

