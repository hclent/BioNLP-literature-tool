from __future__ import print_function
from processors import *
import re, nltk, json, pickle, time, datetime, os.path
from json import JSONEncoder
from nltk.corpus import stopwords

# source activate pyProcessors #my conda python enviornment for this

# set a PROCESSORS_SERVER environment variable.
# It may take a minute or so to load the large model files.
p = '/Users/hclent/anaconda3/envs/py34/lib/python3.4/site-packages/processors/processors-server.jar'
api = ProcessorsAPI(port=8886, jar_path=p, keep_alive=True)
#api.start_server(p)


eng_stopwords = nltk.corpus.stopwords.words('english') #remove default english stopwords 


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
  #clean_text = re.sub('\s{5,50}', '', clean_text) #for anything with 5-20 spaces between them smooth them together. This is intended to help with math
  clean_text = re.sub('\([ATGC]*\)', '', clean_text) #delete any DNA seqs
  clean_text = re.sub('(\(|\)|\'|\]|\[|\\|\,)', '', clean_text) #delete certain stray punctuation
  clean_text = re.sub('\\\\xa0\d*\.?\d?[\,\-]?\d*\,?\d*', '', clean_text) #delete formatting around figures
  clean_text = re.sub('et\sal\.', ' ', clean_text) #replace "et al." with a space
  clean_text = re.sub('\s\d{4}[\s\.\,\;\-]?(\d{4})?', '', clean_text) #delete years
  clean_text = re.sub('[\.\,]\d{1,2}[\,\-]?(\d{1,2})?\,?', '', clean_text) #delete citations
  clean_text = re.sub('Fig\.|Figure', '', clean_text) #delete 'fig.' and 'figure'
  clean_text = clean_text.lower()
  print(clean_text)
  print()
  # print("* Annotating with the Processors ...")
  # print("* THIS MAY TAKE A WHILE ...")
  # biodoc = api.bionlp.annotate(clean_text) #annotates to JSON
  # print("* Successfully did the preprocessing !!!")
  # print("* Dumping JSON ... ")
  # save_path = '/home/hclent/data/' #must save to data
  # completeName = os.path.join(save_path, ('doc_'+(str(pmid))+'_'+str(doc_num)+'.json'))
  # with open(completeName, 'w') as outfile:
  #   json.dump(biodoc, outfile, default=dumper, indent=2)
  # print("* Dumped to JSON !!! ")


#Input: filehandle and max number of documents to process
#Output: JSONified annotated BioDoc 
def loadDocuments(maxNum, pmid):
  print("* Loading dataset...")
  i = 27
  filenamePrefix = "/Users/hclent/Desktop/data_bionlp/"+pmid+"_"
  print(filenamePrefix)
  for i in range(27, int(maxNum)+1):
    print("* Loading document #" + str(i) + " ...")
    filename = filenamePrefix + str(i) + ".txt"
    text = open(filename, 'r')
    text = text.read()
    preProcessing(text, pmid, i)
    i +=1
    print("\n")


# print()
t0 = time.time()
loadDocuments(35, "18952863")
print("annotated docs: done in %0.3fs." % (time.time() - t0))


####################################
#Input: Processors annotated biodocs
#Output: String of lemmas
def grab_lemmas(biodoc):
  lemmas_list = biodoc["lemmas"] #list 
  keep_lemmas = [w for w in lemmas_list if w.lower() not in eng_stopwords]
  keep_lemmas = (' '.join(map(str, keep_lemmas))) #map to string. strings are necessary for the TFIDF
  return keep_lemmas


#Input: Processors annotated biodocs
#Output: List of named entities 
def grab_nes(biodoc):
  ners_list = biodoc["nes"] #list 
  return ners_list

#Input: Processors annotated biodocs (from JSON)
#Output: List of strings of all lemmas 
def loadBioDoc(maxNum, pmid):
  data_samples = []
  nes_list = []
  i = 1
  filenamePrefix = '/Users/hclent/Desktop/data_bionlp/doc_'+(pmid)+'_'
  print(filenamePrefix)
  for i in range(1, maxNum+1):
    print("* Loading annotated BioDoc from JSON #" + str(i) + " ...")
    filename = filenamePrefix + str(i) + ".json"
    with open(filename) as data_file:
      data = json.load(data_file)
      lemmas = grab_lemmas(data)
      data_samples.append(lemmas)
      nes = grab_nes(data)
      nes_list.append(nes)

    i +=1
  return data_samples, nes_list

