from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from processors import *
import re, nltk, json, pickle
from json import JSONEncoder
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.datasets import fetch_20newsgroups
import pprint


# source activate pyProcessors #my conda python enviornment for this

# set a PROCESSORS_SERVER environment variable.
# It may take a minute or so to load the large model files.
#api = ProcessorsAPI(port=8886)
#api.start_server("/Users/hclent/anaconda3/envs/pyProcessors/lib/python3.4/site-packages/processors/processors-server.jar")


eng_stopwords = nltk.corpus.stopwords.words('english') #remove default english stopwords 
bio_stopwords = ['et', 'al', 'fig', 'author'] #add hand picked bio stopwords to stopwords
for word in bio_stopwords:
	eng_stopwords.append(word)


#Input: Data that you want to be JSONified
#Output: Data reformatted so it can be dumped to JSON
def dumper(obj):
  try:
    return obj.toJSON()
  except:
    return obj.__dict__

#Input: String(text), pmid, doc_num
#Output: This method cleans the text of newline markups, DNA sequences, and some punctuation
#Output: Then it makes a "biodoc" using the PyProcessor's "BioNLP" Processor. This step takes a while for longer docs
#Output: This doc is saved to JSON. 
#Output: pmid and doc_num are for naming the JSON filename 
def preProcessing(text, pmid, doc_num):
  print("* Preprocessing the text ... ")
  clean_text = re.sub('\\\\n', ' ', text) #replace \n with a space
  clean_text = re.sub('\([ATGC]*\)', '', clean_text) #delete any DNA seqs
  clean_text = re.sub('(\(|\)|\'|\]|\[|\\|\,)', '', clean_text) #delete certain punctuation
  clean_text = clean_text.lower()
  print("* Annotating with the Processors ...")
  print("* THIS MAY TAKE A WHILE ...")
  biodoc = api.bionlp.annotate(clean_text) #annotates to JSON
  print("* Successfully did the preprocessing !!!")
  print("* Dumping JSON ... ")
  with open('json_'+(str(pmid))+'_'+str(doc_num)+'.json', 'w') as outfile:
    json.dump(biodoc, outfile, default=dumper, indent=2)
  print("* Dumped to JSON !!! ")


#Input: filehandle and max number of documents to process
#Output: JSONified annotated BioDoc 
def loadDocuments(filenamePrefix, maxNum):
  print("* Loading dataset...")
  i = 1
  for i in range(1, maxNum+1):
    print("* Loading document #" + str(i) + " ...")
    filename = filenamePrefix + str(i) + ".txt"
    text = open(filename, 'r')
    text = text.read()
    preProcessing(text,18952863,i)
    i +=1
    print("\n")

prefix = "/Users/hclent/Desktop/BioNLP-literature-tool/1_info_retrieval/18952863_"
#loadDocuments(prefix, 84)


def grab_lemmas(biodoc):
  lemmas_list = biodoc["lemmas"] #list 
  keep_lemmas = [w for w in lemmas_list if w.lower() not in eng_stopwords]
  keep_lemmas = (' '.join(map(str, keep_lemmas))) #map to string. strings are necessary for the TFIDF
  return keep_lemmas

def grab_nes(biodoc):
  ners_list = biodoc["nes"] #list 
  return ners_list


def loadBioDoc(filenamePrefix, maxNum):
  data_samples = []
  i = 1
  for i in range(1, maxNum+1):
    #print("* Loading annotated BioDoc from JSON #" + str(i) + " ...")
    filename = filenamePrefix + str(i) + ".json"
    with open(filename) as data_file:
      data = json.load(data_file)
      lemmas = grab_lemmas(data)
      data_samples.append(lemmas)
    i +=1
  return data_samples


#json_prefix = '/Users/hclent/Desktop/BioNLP-literature-tool/3_cluster/json_18952863_'
#data_samples = loadBioDoc(json_prefix, 84)
#pickle.dump(data_samples, open("18952863_all.p", "wb"))
############################# LDA ############################################
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
  lda = LatentDirichletAllocation(n_topics=5, max_iter=5, learning_method='online', learning_offset=50., random_state=1)
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
print(tfidf)
#lda = fit_lda(tfidf)
#print("############## RESULTS ###############")
#topics_lda(tfidf_vectorizer, lda)
