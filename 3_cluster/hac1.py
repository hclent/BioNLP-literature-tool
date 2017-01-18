from __future__ import print_function
from sklearn import feature_extraction
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import pickle, time
import numpy as np



#list of strings
data_samples = pickle.load(open("/home/hclent/data/data_samples/data_samples_18952863+18269575+21364914.pickle", "rb"))


#Input: Eata_samples (list of lists containing strings)
#Output: Sparse matrix, l2 normalization for preserving Euclidean distance
def get_hashing(data):
  t0 = time.time()
  print("* Making hashing vectorizor with the data ...")
  hasher = HashingVectorizer(stop_words='english', ngram_range=(1,3), norm='l2', non_negative=True) #l2 projected on the euclidean unit sphere
  hX = hasher.fit_transform(data)
  print("done in %0.3fs." % (time.time() - t0))
  dense_hX = hX.toarray()
  return dense_hX


def get_tfidf(data): #data should be a list of strings for the documents
  print("* Making tfidf with the data ...")
  tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), norm='l2') #l2 projected on the euclidean unit sphere
  tfidf = tfidf_vectorizer.fit_transform(data)
  dense_tfidf = tfidf.toarray()
  return dense_tfidf

# tfidfX, tfidf_vectorizer = get_tfidf(data_samples)

#Input: High dimensional (sparse) matrix
#But HAC needs a dense matrix
#Output: Clusters
def do_hac(dense_matrix):
    t0 = time.time()
    print("* Beginning HA clustering ... ")
    n_clusters = 3
    hac = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    #dense = sparse_matrix.toarray()
    hac.fit(dense_matrix)
    label = hac.labels_
    print(label)
    print("done in %0.3fs." % (time.time() - t0))

#dense_hX = get_hashing(data_samples)
dense_tfidf = get_tfidf(data_samples)
print(type(dense_tfidf))
do_hac(dense_tfidf)
