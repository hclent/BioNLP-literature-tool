from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import sys, pickle, math, random, numpy

data_samples = pickle.load(open("18952863_all.p", "rb")) #pre-processed already 

#Input: Eata_samples (list of lists containing strings)
#Output: Sparse matrix, l2 normalization for preserving Euclidean distance
def get_hashing(data):
  print("* Making hashing vectorizor with the data ...")
  hasher = HashingVectorizer(stop_words='english', ngram_range=(1,3), norm='l2', non_negative=False) #l2 projected on the euclidean unit sphere
  hX = hasher.fit_transform(data)
  return hX, hasher

#Input: High dimensional matrix
#Output: Centroids
def do_kemeans(sparse_matrix):
    num_clusters = 3
    km = KMeans(init='k-means++', n_clusters=num_clusters)
    km.fit(sparse_matrix)
    clusters = km.labels_.tolist()
    return km, clusters


#Truncated SVD (LSA) for dimensionality reduction
#For plotting
#Try PCA
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


kmclusters_H = do_kemeans(hX) #list of cluster assignments
print("K-means with Hashing Vectors ... ")
print(kmclusters_H)
print()


hX, hasher = get_hashing(data_samples)
print(hX.toarray())
print(hX.shape)
print()


svdX = dimen_reduce(hX)
print(svdX)
print(svdX.shape) #(84, 3)




####### GRAVEYARD ##########
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_

#Hashing Vector is better than TF-IDF for K-means because of Eucliean distance

# def get_tfidf(data): #data should be a list of strings for the documents 
#   print("* Making tfidf with the data ...")
#   tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), norm='l2') #l2 projected on the euclidean unit sphere
#   tfidf = tfidf_vectorizer.fit_transform(data)
#   return tfidf, tfidf_vectorizer

# tfidfX, tfidf_vectorizer = get_tfidf(data_samples)
# print(tfidfX.toarray())
# print(tfidfX.shape)
# print()
