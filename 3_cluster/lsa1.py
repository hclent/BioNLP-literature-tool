from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
import sys, pickle, math, random 


data_samples = pickle.load(open("18952863_all.p", "rb")) #pre-processed 


def get_tfidf(data): #data should be a list of strings for the documents 
  tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2, 3), norm='l2')
  tfidf = tfidf_vectorizer.fit_transform(data) #tfidf is sparse matrix 
  return tfidf, tfidf_vectorizer


def do_LSA(X, vectorizer): #where X is a tfidf matrix (sparse)
  lsa = TruncatedSVD(n_components=5, n_iter=100)
  lsa_results = lsa.fit(X)
  #print(lsa_results.components_[0]) # V matrix [m x k]^Transposed, rows = terms, columns = concepts
  #[ 0.00328614  0.00047173  0.00047173 ...,  0.0002086   0.0002086   0.0002086 ]
  terms = vectorizer.get_feature_names()
  for i, comp in enumerate(lsa.components_):
    termsInComp = zip(terms, comp)
    sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:10]
    print("Concept "+ str(i) + ": ")
    for term in sortedTerms:
      print(term[0])
    print()

tfidf, tfidf_vectorizer = get_tfidf(data_samples)
#print(tfidf.shape) #(84 x 184,867) with bigrams, (84, 431901) with trigrams 
do_LSA(tfidf, tfidf_vectorizer)