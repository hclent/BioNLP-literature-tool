from __future__ import print_function
from sklearn import feature_extraction
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import pickle, time
import numpy as np
import pandas as pd
import nltk, re, os, codecs
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity


#list of strings
data_samples = pickle.load(open("/home/hclent/data/data_samples/data_samples_18952863+18269575+21364914.pickle", "rb"))

#global
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

## These are too slow for my liking :/
def tokenize_only(data_samples):
    print("tokenize only")
    tokenized_data = []
    for text in data_samples:
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        tokenized_data.extend(filtered_tokens)
    return tokenized_data

def tokenize_and_stem(data_samples):
    print("tokenize and stem")
    stemmed_data = []
    for text in data_samples:
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]
        stemmed_data.extend(stems)
    return stemmed_data


def get_tfidf(data_samples):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.2, stop_words='english',
                                       use_idf=True, ngram_range=(1, 3)) #tokenizer=tokenize_and_stem
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_samples)  # fit the vectorizer to synopses
    print(tfidf_matrix.shape)
    terms = tfidf_vectorizer.get_feature_names()
    #dist = 1 - cosine_similarity(tfidf_matrix)
    return tfidf_matrix, terms


#Input: High dimensional (sparse) matrix
#Output: Clusters
# labels = km.labels_
# centroids = km.cluster_centers_
def do_kemeans(sparse_matrix):
    t0 = time.time()
    print("* Beginning k-means clustering ... ")
    num_clusters = 3
    km = KMeans(init='k-means++', n_clusters=num_clusters)
    km.fit(sparse_matrix)
    clusters = km.labels_.tolist()
    print("done in %0.3fs." % (time.time() - t0))
    return clusters, km



totalvocab_tokenized = tokenize_only(data_samples[0:5]) #610014
totalvocab_stemmed = tokenize_and_stem(data_samples[0:5]) #610014
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print(vocab_frame.head())

test_docs = data_samples[0:5]
titles = ['a', 'b', 'c', 'd', 'e']
ranks = [0, 1, 2, 3, 4]
tfidf, terms = get_tfidf(data_samples[0:5]) #not tokenized and stemmed fuq
clusters, km = do_kemeans(tfidf) #list of cluster assignments
genres = ['sci', 'sci', 'bio', 'genetics', 'trash']
docs = { 'title': titles, 'rank': ranks, 'synopsis': test_docs, 'cluster': clusters, 'genre': genres }
frame = pd.DataFrame(docs, index = [clusters] , columns = ['rank', 'title', 'cluster', 'genre'])
print(frame)
print(frame['cluster'].value_counts()) #number of films per cluster (clusters from 0 to 4)
grouped = frame['rank'].groupby(frame['cluster']) #groupby cluster for aggregation purposes

print("Top terms per cluster:")
print()
# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
num_clusters = 3
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :12]:  # replace 12 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')
    print()  # add whitespace
    print()  # add whitespace


    '''
On a set of 5 test documents the results for k=3 are:

Cluster 0 words: nan, nan, bac, lincrna, atlincrna, nan,

Cluster 1 words: nan, nan, nan, gene, nan, nan,

Cluster 2 words: nan, breakpoint, scaffolding, number, tell, nan,

Maybe this method could be made to be more efficient, but
    1) stemming and tokenizing should be required in tfidf step but instead makes it return only single letters. But,
       don't use it and you get nan
    2) stemming and tokenizing, then tokenizing sepperately? barf

    '''
