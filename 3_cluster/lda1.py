from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re, json, pickle
import pprint


data_samples = pickle.load(open("18952863_all.p", "rb"))


def get_tfidf(data): #data should be a list of strings for the documents 
  print("* Preparing to vectorize data ...")
  tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), norm='l2')
  print("* Fitting data to vector ...")
  tfidf = tfidf_vectorizer.fit_transform(data)
  print("* Successfully fit data to the vector !!! ")
  return tfidf, tfidf_vectorizer


def fit_lda(tfidf):
  print("* Initializing Latent Dirichlet Allocation ... ")
  lda = LatentDirichletAllocation(n_topics=3, max_iter=25, learning_method='online', learning_offset=50., random_state=1)
  lda.fit(tfidf)
  print("* Successfully fit data to the model!!! ")
  return lda


def print_top_words(model, feature_names, n_top_words):
  for topic_idx, topic in enumerate(model.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


def topics_lda(tf_vectorizer, lda):
  print("\nTopics in LDA model:")
  tf_feature_names = tf_vectorizer.get_feature_names()
  results = print_top_words(lda, tf_feature_names, 6)
  #print(tf_feature_names)


tfidf, tfidf_vectorizer = get_tfidf(data_samples)
#print(tfidf)
lda = fit_lda(tfidf)
#print("############## RESULTS ###############")
topics_lda(tfidf_vectorizer, lda)
