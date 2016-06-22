from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import scale
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from scipy import cluster
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sys, pickle, math, random 
from collections import defaultdict

data_samples = pickle.load(open("18952863_all.p", "rb")) #pre-processed 


def get_tfidf(data): #data should be a list of strings for the documents 
  print("Making tfidf with the data ...")
  tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), norm='l2')
  print("make the vectorizor ...")
  tfidf = tfidf_vectorizer.fit_transform(data) #tfidf is sparse matrix 
  print("fit the X ...")
  return tfidf, tfidf_vectorizer

tfidf, tfidf_vectorizer = get_tfidf(data_samples)
#print(tfidf)
print(tfidf.shape) 


def do_kemeans(sparse_matrix):
    num_clusters = 3
    km = KMeans(init='k-means++', n_clusters=num_clusters)
    km.fit(sparse_matrix)
    clusters = km.labels_.tolist()
    return clusters


# np.random.seed(123)
# tests = np.reshape( np.random.uniform(0,100,60), (30,2) )
# #print(tests[1:4])

# #plot variance for each value for 'k' between 1,10
# initial = [cluster.vq.kmeans(tests,i) for i in range(1,10)]
# # plt.plot([var for (cent,var) in initial])
# # plt.show()

# cent, var = initial[3]
# #use vq() to get as assignment for each obs.
# assignment,cdist = cluster.vq.vq(tests,cent)
# plt.scatter(tests[:,0], tests[:,1], c=assignment)
# plt.show()

kmclusters = do_kemeans(tfidf) #list of cluster assignments
print(kmclusters)


# doc_names = ['PMC4895484', 'PMC4867222', 'PMC4851370', 'PMC4832253', 
# 'PMC4811886', 'PMC4810023', 'PMC4795695', 'PMC4783527', 'PMC4783356', 'PMC4732000', 
# 'PMC4726258', 'PMC4679983', 'PMC4646352', 'PMC4603330', 'PMC4548842', 'PMC4464145', 'PMC4457800', 
# 'PMC4404975', 'PMC4397884', 'PMC4384038', 'PMC4383980', 'PMC4315028', 'PMC4308911', 'PMC4295026', 
# 'PMC4289383', 'PMC4256881', 'PMC4240082', 'PMC4239561', 'PMC4239027', 'PMC4145113', 'PMC4119204', 
# 'PMC4112214', 'PMC4109783', 'PMC4102449', 'PMC4071528', 'PMC4038794', 'PMC4034038', 'PMC4013592', 
# 'PMC3970191', 'PMC3968010', 'PMC3967086', 'PMC3961402', 'PMC3958358', 'PMC3876103', 'PMC3866978', 
# 'PMC3852042', 'PMC3851865', 'PMC3850447', 'PMC3817805', 'PMC3751884', 'PMC3590777', 'PMC3576282', 
# 'PMC3531202', 'PMC3491363', 'PMC3481203', 'PMC3474763', 'PMC3439424', 'PMC3436819', 'PMC3430884', 
# 'PMC3430549', 'PMC3408644', 'PMC3406906', 'PMC3389459', 'PMC3383449', 'PMC3358895', 'PMC3355796', 
# 'PMC3355756', 'PMC3326336', 'PMC3280228', 'PMC3271752', 'PMC3269863', 'PMC3240960', 'PMC3219971', 
# 'PMC3203428', 'PMC3152334', 'PMC3095355', 'PMC3053395', 'PMC3049755', 'PMC3045888', 'PMC2924827', 
# 'PMC2893956', 'PMC2886086', 'PMC2853686', 'PMC2817432']

# def makeDict(clusters, labels):
#   combined = list(zip(labels, clusters))
#   c0_Dict = defaultdict(lambda: 0, size = 743)
#   c1_Dict = defaultdict(lambda: 1)
#   c2_Dict = defaultdict(lambda: 2)
#   for key, values in combined:
#     if values == 0:
#       c0_Dict[key] == values
#     if values == 1:
#       c1_Dict[key] == values
#     if values == 2:
#       c2_Dict[key] == values
#   return c0_Dict, c1_Dict, c2_Dict


