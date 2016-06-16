# BioNLP-literature-tool

Under construction 

Pipeline: information retrieval --> natural language processing --> topic modeling (clustering) --> data visualization --> web interface 

A Google Doc presentation of my pipeline can be found here:
https://docs.google.com/presentation/d/1zntsTeNRg0tUBChUQBCjK-PbOZamWvaaANVdTegBQNU/edit?usp=sharing


#Information retrieval#
Entrez_IR.py in '1_info_retrieval' takes a pubmed ID and output text files of the publications citing the original pubmedID. This code uses NCBI's API. All citing documents are written to a txt file to undergo nlp.

#NLP#
Scripts in 'nlp' are for tokenization, POS tagging, ngrams, and feature based named entity recognition (NER). 

Note: Current feature based NER is being phased out in favor of using Clulab's Scala-based BioNLP NER tool.


#Topic Modeling / Clustering#
Switching to Latent Dirichlet Allocation (LDA) machine learning algorithm for document clustering. The topics here, in combination with the NERs obtained from NLP, will be used for data visualization.

#Data Visualization#
In progress... 

#Web Interface#
Please see my repository called "Webdev for bioNLP lit tool"
https://github.com/hclent/Webdev-for-bioNLP-lit-tool

#Graveyard#
This is where I am keeping depreciated code that I am no longer using for my project but that I think is still valuable for the github community :) 

