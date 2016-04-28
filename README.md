# BioNLP-literature-tool

Under construction 

Pipeline: crawler --> natural language processing --> classification --> visualization --> web interface 

#Crawler#
Scripts under 'crawler' take a pubmed ID and then crawls PMC in order to scrape the publictions citing the original pubmid ID document. All citing documents are written to a txt file to undergo nlp.

#NLP#
Scripts in 'nlp' are for tokenization, POS tagging, ngrams, and feature based named entity recognition (NER). Linguistics elements will be used as features for the classifer, and the NER will also be used for visualization.

#Classification#
Scripts here will be for document classification. Currently, the only script posted looks at cosine similarity as a feature for the classifier. 

#Visualization#
Will post soon

#Web Interface#
Will post soon
