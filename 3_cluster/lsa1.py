from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
import pickle, re 

data_samples = pickle.load(open("18952863_all.p", "rb")) #pre-processed 

#Input: a list of strings for the document
#Output: TFIDF matrix X, and TfidfVectorizer function
#Output: TFIDF matrix X is a sparse matrix
def get_tfidf(data): 
  tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2, 3), norm='l2')
  tfidf = tfidf_vectorizer.fit_transform(data) 
  return tfidf, tfidf_vectorizer


#Input: TFIDF matrix (X) and TfidfVectorizer
#Output: json dict ready for D3 visualization
#Output: this dict has Latent Semantic Analysis topics (concepts) and topic words
#To Do: Use dimensionality reduction for something... 
def do_LSA(X, vectorizer): 
  lsa = TruncatedSVD(n_components=3, n_iter=100)
  lsa_results = lsa.fit(X)
  #dimen_reduc = lsa.transform(X)
  #print(dimen_reduc.shape) (84, 3)
  terms = vectorizer.get_feature_names()
  jDict = {"name": "flare", "children": []} #initialize dict for json
  for i, comp in enumerate(lsa.components_):
    #print(lsa_results.components_[0]) # V matrix [m x k]^Transposed, rows = terms, columns = concepts
    #[ 0.00328614  0.00047173  0.00047173 ...,  0.0002086   0.0002086   0.0002086 ]
    termsInComp = zip(terms, comp)
    sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:10]
    #print("Concept "+ str(i) + ": ")
    running_name = 'concept'+str(i)
    concept_Dict = {"name": running_name, "children": []}
    jDict["children"].append(concept_Dict)
    for term in sortedTerms:
      #print(term[0])
      term_Dict = {"name": term[0], "size": 700}
      concept_Dict["children"].append(term_Dict)
  jsonDict = re.sub('\'', '\"', str(jDict)) #json needs double quotes, not single quotes
  return jsonDict

tfidf, tfidf_vectorizer = get_tfidf(data_samples)
#print(tfidf.shape) #(84 x 184,867) with bigrams, (84, 431901) with trigrams 

jsonDict = do_LSA(tfidf, tfidf_vectorizer)
print(jsonDict)

