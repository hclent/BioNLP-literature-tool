import gensim, pickle, string, time
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
#
# where doc_clean is [doc, doc, doc]
t0 = time.time()
print("accessing data ... ")
data_samples = pickle.load(open("/home/hclent/data/data_samples/data_samples_18952863+18269575+21364914.pickle", "rb"))
print(type(data_samples[0]))
print(data_samples[0])
#stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    #stop_free = " ".join([i for i in doc.lower().split() if i not in stop]) #stop words already out
    punc_free = ''.join(ch for ch in doc if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
print("cleaning data ... ")
doc_clean = [clean(doc).split() for doc in data_samples]
print("making corpus dictionary ... ")
dictionary = corpora.Dictionary(doc_clean)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
print("making doc_term_matrix .... ")
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
print("done with doc_term_matrix .... now beginning LDA")
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
print("defined LDA model...")
print("making LDA model....")
# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=6, id2word = dictionary, passes=1)
print("topics:")
blah = (ldamodel.print_topics(num_topics=6, num_words=10))
for b in blah:
    print(b)
print("done in %0.3fs." % (time.time() - t0))


