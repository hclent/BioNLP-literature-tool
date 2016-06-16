from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pandas as pd
import sys
import pickle


data_samples = pickle.load(open("18952863_all.p", "rb"))

def get_tfidf(data): #data should be a list of strings for the documents 
  print("Making tfidf with the data ...")
  tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), norm='l2')
  print("make the vectorizor")
  tfidf = tfidf_vectorizer.fit_transform(data)
  print("fit the X")
  return tfidf, tfidf_vectorizer

tfidf, tfidf_vectorizer = get_tfidf(data_samples)
print(tfidf.shape)



num_clusters = 5
km = KMeans(n_clusters = num_clusters)
km.fit(tfidf)
clusters = km.labels_.tolist()
print(clusters)


