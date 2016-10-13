import naive_makeVecs as makeVecs
import pickle


#Load from pickled data_samples instead of filename
def loadFromDataSamples(data_samples):
    vecs_list = []

    for document in data_samples:
        vectorCounter = makeVecs.text2vec(document)
        #print(vectorCounter)
        vecs_list.append(vectorCounter)
    return vecs_list


#Load txt file
def loadMessages(filename):
    fcorpus = open(filename, 'r')
    fcorpus = fcorpus.read() #str

    vectorCounter = makeVecs.text2vec(fcorpus)

    return (vectorCounter)



# Print cosine similarity scores
def cosineSimilarityScore(vector1, vector2):

    # cosine_sim_score1 = (str(makeVecs.cosine(vector1, vector1)))
    # print("cos (vec1, vec1): " + cosine_sim_score1)
    #
    #
    # cosine_sim_score2 = (str(makeVecs.cosine(vector2, vector2)))
    # print("cos (vec2, vec2): " + cosine_sim_score2)


    cosine_sim_score_1_2 = (str(makeVecs.cosine(vector1, vector2)))
    print("cos (vec1, vec2): " + cosine_sim_score_1_2)
    print("-------------------------------------------------------")
    return cosine_sim_score_1_2



star_trek = "/home/hclent/data/corpora/startrek/105.txt"
vecs1 = loadMessages(star_trek)

# doc2 = "/home/hclent/data/18269575/18269575_1.txt"
# doc3 = "/home/hclent/data/18269575/18269575_2.txt"
# vecs2 = loadMessages(doc2)
# vecs3= loadMessages(doc3)


data_samples = pickle.load(open("/home/hclent/data/18269575/data_samples_18269575.pickle", "rb")) #pre-processed

vecs_list = loadFromDataSamples(data_samples)
cosine_list = []
for vec_n in vecs_list:
    cosine_sim_score_1_2 = cosineSimilarityScore(vecs1, vec_n)
    cosine_list.append(cosine_sim_score_1_2)


print(cosine_list)


############### EXAMPLE #################
# cos (star_trek, star_trek): 0.9999999999999648 #self
# cos (coge1, coge1): 0.9999999999999903  #self
# cos (star_trek, coge1): 0.07009289426452932
# cos (star_trek, coge2): 0.029406530815802755
# cos (coge1, coge2): 0.6057148741444401

