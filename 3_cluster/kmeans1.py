from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import sys, pickle, math, random, numpy
import plotly.plotly as py
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_samples = pickle.load(open("18952863_all.p", "rb")) #pre-processed already 


def get_tfidf(data): #data should be a list of strings for the documents 
  print("* Making tfidf with the data ...")
  tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), norm='l2') #l2 projected on the euclidean unit sphere
  tfidf = tfidf_vectorizer.fit_transform(data)
  return tfidf, tfidf_vectorizer

#tfidfX, tfidf_vectorizer = get_tfidf(data_samples)
# print(tfidfX.toarray())
# print(tfidfX.shape)
# print()

def get_hashing(data):
  print("* Making hashing vectorizor with the data ...")
  hasher = HashingVectorizer(stop_words='english', ngram_range=(1,3), norm='l2', non_negative=False) #l2 projected on the euclidean unit sphere
  hX = hasher.fit_transform(data)
  return hX, hasher


hX, hasher = get_hashing(data_samples)
# print(hX.toarray())
# print(hX.shape)
# print()

#Truncated SVD (LSA) for dimensionality reduction
def dimen_reduce(sparse_matrix):
  print("* Performing SVD on sparse matrix ... ")
  svd = TruncatedSVD(n_components=3, n_iter=100)
  normalizer = Normalizer(copy=False)
  lsa = make_pipeline(svd, normalizer)
  X = lsa.fit_transform(sparse_matrix)
  explained_variance = svd.explained_variance_ratio_.sum()
  print("Explained variance of the SVD step: {}%".format(
    int(explained_variance * 100)))
  return X

svdX = dimen_reduce(hX)
print(svdX)
# print(svdX.shape) #(84, 3)



def do_kemeans(sparse_matrix):
    num_clusters = 3
    km = KMeans(init='k-means++', n_clusters=num_clusters)
    km.fit(sparse_matrix)
    clusters = km.labels_.tolist()
    return km, clusters


# kmclusters_T = do_kemeans(tfidfX) #list of cluster assignments
# print("K-means with TF-IDF ... ")
# print(kmclusters_T)
# print()

# kmclusters_H = do_kemeans(hX) #list of cluster assignments
# print("K-means with Hashing Vectors ... ")
# print(kmclusters_H)
# print()

kmeans, kmclusters_L = do_kemeans(svdX) #list of cluster assignments
print("K-means with Dimensionality-Reduced Vectors ... ")
print(kmclusters_L)
print()


##############################################################
scatter = dict(
    mode = "markers",
    name = "y",
    type = "scatter3d",    
    x = kmclusters_L['x'], y = kmclusters_L['y'], z = kmclusters_L['z'],
    marker = dict(size=2, color="rgb(23, 190, 207)"))

clusters = dict(
    alphahull = 7,
    name = "y",
    opacity = 0.1,
    type = "mesh3d",    
    x = svdX['x'], y = svdX['y'], z = svdX['z'])

layout = dict(
    title = '3d point clustering',
    scene = dict(
        xaxis = dict( zeroline=False ),
        yaxis = dict( zeroline=False ),
        zaxis = dict( zeroline=False )))

fig = dict( data=[scatter, clusters], layout=layout )
py.iplot(fig, filename='3d point clustering')


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