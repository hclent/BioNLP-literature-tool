from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from processors import *
import re, nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# source activate pyProcessors #my conda python enviornment

# set a PROCESSORS_SERVER environment variable.
# It may take a minute or so to load the large model files.
#api = ProcessorsAPI(port=8886)


print("Loading dataset...")
prefix = "/Users/hclent/Desktop/BioNLP-literature-tool/1_info_retrieval/18952863_"

eng_stopwords = nltk.corpus.stopwords.words('english') #remove default english stopwords 
# bio_stopwords = ['et', 'al', 'fig', 'author'] #add hand picked bio stopwords to stopwords
# for word in bio_stopwords:
# 	eng_stopwords.append(word)

def preProcessing(text):
	clean_text = re.sub('\\\\n', ' ', text) #replace \n with a space
	clean_text = re.sub('\([ATGC]*\)', '', clean_text) #delete any DNA seqs
	clean_text = re.sub('(\(|\)|\'|\]|\[|\\|\,)', '', clean_text) #delete certain punctuation
	clean_text = clean_text.lower()
	biodoc = api.bionlp.annotate(clean_text) #str
	lemmas_list = biodoc.lemmas #list 
	keep_lemmas = [w for w in lemmas_list if w.lower() not in eng_stopwords]
	preprocessed_text = (' '.join(map(str, keep_lemmas))) #map to string. strings are necessary for the TFIDF
	return preprocessed_text
  #t0 = time()
	#print("done in %0.3fs." % (time() - t0))


def loadDocuments(filenamePrefix, maxNum):
  data_samples = [] #need a list of docs as strings
  i = 1
  for i in range(1, maxNum+1):
    filename = filenamePrefix + str(i) + ".txt"
    text = open(filename, 'r')
    text = text.read()
    words = preProcessing(text)
    data_samples.append(words)
    i +=1
  return data_samples
#
#data_samples = loadDocuments(prefix, 1) #API connection issues when looping through docs



################ LDA ############################################
doc1 = ['pineapple ananas comosus l. merr. is the most economically', 
'valuable crop possessing crassulacean acid metabolism cam, a photosynthetic carbon assimilation pathway with high water use efficiency',
'and the second most important tropical fruit after banana in terms of international trade. we sequenced the genomes of pineapple varieties ‘f153’ and ‘md2’, and a wild pineapple relative a. bracteatus accession cb5. the pineapple genome has one fewer ancient whole genome duplications than sequenced grass genomes and, therefore, provides an important reference for elucidating gene content and structure in the last common ancestor of extant members of the grass family poaceae.',
'pineapple has a conserved karyotype with seven pre rho duplication chromosomes that are ancestral to extant grass karyotypes. the pineapple lineage has transitioned from c3 photosynthesis to cam with cam-related genes exhibiting a diel expression pattern in photosynthetic tissues using beta-carbonic anhydrase βca for initial capture of co2. promoter regions of all three βca genes contain a cca1 binding site', 
'that can bind circadian core oscillators. cam pathway genes were enriched with cis-regulatory elements including the morning  and evening  elements associated with regulation of circadian-clock genes, providing the first link between cam and the circadian clock regulation. gene-interaction network analysis revealed both activation and repression of regulatory elements that control key enzymes in cam photosynthesis', 
'indicating that cam evolved by reconfiguration of pathways preexisting in c3 plants. pineapple cam photosynthesis is the result of regulatory neofunctionalization of preexisting gene copies and not acquisition of neofunctionalized genes via whole genome or tandem gene duplication.       , these authors contributed equally to this work., pineapple ananas comosus l. merr. is the most economically valuable crop possessing crassulacean acid metabolism cam, a photosynthetic carbon assimilation pathway',
'with high water use efficiency, and the second most important tropical fruit after banana in terms of international trade. we sequenced the genomes of pineapple varieties ‘f153’ and ‘md2’, and a wild pineapple relative a. bracteatus accession cb5. the pineapple genome has one fewer ancient whole genome duplications than sequenced grass genomes and, therefore, provides an important reference for elucidating gene content and structure in the last common ancestor of extant members of the grass family poaceae.',
'pineapple has a conserved karyotype with seven pre rho duplication chromosomes that are ancestral to extant grass karyotypes. the pineapple lineage has transitioned from c3 photosynthesis to cam with cam-related genes exhibiting a diel expression pattern in photosynthetic tissues using beta-carbonic anhydrase βca for initial capture of co2. promoter regions of all three βca genes contain a cca1 binding site that can bind circadian core oscillators. cam pathway genes were enriched with cis-regulatory elements including the morning  and evening  elements associated with regulation of circadian-clock genes, providing the first link between cam and the circadian clock regulation. gene-interaction network analysis revealed both activation and repression of regulatory elements that control key enzymes in cam photosynthesis, indicating that cam evolved by reconfiguration of pathways preexisting in c3 plants. pineapple cam photosynthesis is the result of regulatory neofunctionalization of preexisting gene copies and not acquisition of neofunctionalized genes via whole genome or tandem gene duplication.']

def get_tfidf(data): #data should be a list of strings for the documents 
  print("Making tfidf with the data ...")
  tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), norm='l2')
  print("make the vectorizor")
  tfidf = tfidf_vectorizer.fit_transform(data)
  print("fit the X")
  return tfidf, tfidf_vectorizer

def fit_lda(tfidf):
  lda = LatentDirichletAllocation(n_topics=5, max_iter=5, learning_method='online', learning_offset=50., random_state=0)
  t0 = time()
  lda.fit(tfidf)
  return lda
  print("done in %0.3fs." % (time() - t0))


def print_top_words(model, feature_names, n_top_words):
  for topic_idx, topic in enumerate(model.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
  print()


def topics_lda(tf_vectorizer, lda):
  print("\nTopics in LDA model:")
  tf_feature_names = tf_vectorizer.get_feature_names()
  results = print_top_words(lda, tf_feature_names, 3)
  #print(tf_feature_names)
  print(results)


tfidf, tfidf_vectorizer = get_tfidf(doc1)
print(tfidf)
lda = fit_lda(tfidf)
#print(lda)
topics_lda(tfidf_vectorizer, lda)
