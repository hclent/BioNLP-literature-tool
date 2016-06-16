import makeVecs as makeVecs
import math
import string
import re
from collections import Counter


#Input: publication
#Output: normalized vector (counter), representing stringIn.

#Example
doc1 = "7.txt"
doc2 = "92.txt"
movie_review = "movie.txt"


def loadMessages(filename):
    fcorpus = open(filename, 'r')
    fcorpus = fcorpus.read() #str

    vectorCounter = makeVecs.text2vec(fcorpus)

    return (vectorCounter)


vecs1 = loadMessages(doc1)
vecs2 = loadMessages(doc2)
vecs3= loadMessages(movie_review)



#Print cosine similarity scores
def cosineSimilarityScore(vector1, vector2, vector3):

    cosine_sim_score1 = (str(makeVecs.cosine(vector1, vector1)))
    print("cos (vec1, vec1): " + cosine_sim_score1)


    cosine_sim_score2 = (str(makeVecs.cosine(vector2, vector2)))
    print("cos (vec2, vec2): " + cosine_sim_score2)


    cosine_sim_score_1_2 = (str(makeVecs.cosine(vector1, vector2)))
    print("cos (vec1, vec2): " + cosine_sim_score_1_2)


    cosine_sim_score_1_3 = (str(makeVecs.cosine(vector1, vector3)))
    print("cos (vec1, vec3): " + cosine_sim_score_1_3)


    cosine_sim_score_2_3 = (str(makeVecs.cosine(vector2, vector3)))
    print("cos (vec2, vec3): " + cosine_sim_score_2_3)


cosineSimilarityScore(vecs1, vecs2, vecs3)



############### EXAMPLE #################
# cos (vec1, vec1): 0.9999999999999983 #self
# cos (vec2, vec2): 1.0000000000000064 #self
# cos (vec1, vec2): 0.21345131811302565 #two scientific texts
# cos (vec1, vec3): 0.03610515107865741 #scientific versus movie review
# cos (vec2, vec3): 0.03499071664127394 #scientific versus movie review
