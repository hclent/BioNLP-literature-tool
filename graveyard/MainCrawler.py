import requests
from bs4 import BeautifulSoup
import re
import sys
import time

## Run in Python3 for handling unicode!
## Input: PMC 'cited by' page
## Output: Retrieves all article titles, authors, url, journal of publication, and document text


def pmc_spider(max_pages, pmid): #Main spider
	start = 1

	titles_list = []
	url_list = []
	url_keys = []
	authors_list = []

	while start <= max_pages:
		print ("* Beginning crawl ....")
		print ("* Crawling page "+ str(start) + " ...")
		print ("* Retrieving information ...")

		url = 'http://www.ncbi.nlm.nih.gov/pmc/articles/pmid/'+str(pmid)+'/citedby/?page='+str(start)

		req = requests.get(url)
		plain_text = req.text
		soup = BeautifulSoup(plain_text, "lxml")

		for items in soup.findAll('div', {'class': 'title'}):
			title = items.get_text()
			titles_list.append(title)

			for link in items.findAll('a'):
				urlkey = link.get('href')
				url_keys.append(urlkey)   #url = base + key
				url =  "http://www.ncbi.nlm.nih.gov"+str(urlkey)
				url_list.append(url)

		for citation in soup.findAll('div', {'class': 'desc'}):
			author = citation.string
			authors_list.append(author)

		start += 1
	return titles_list, url_list, authors_list


titles_list, url_list, authors_list = pmc_spider(1, '10592259') #look at first n pages
url_count = (len(url_list))

#Finding and Comparing Syntenic Regions among Arabidopsis
#pmid = 18952863

#How to usefully compare homologous plant genes and chromosomes as DNA sequences.
#pmid = 18269575


def get_text(list_paper_urls): #Get publication text
	print ("* Retrieving papers .... this will take a while .... ")

	i = 1
	academic_journals = []

	for paper_url in list_paper_urls:
		print("* accessing paper from......")
		print(paper_url)
		main_text = []

		print("* getting requests .... ")
		req = requests.get(paper_url)

		print("* request obtained .... ")
		plain_text = req.text
		soup = BeautifulSoup(plain_text, 'lxml')

		# Get journal of publication #Slow
		for journals in soup.findAll('a', class_ = 'navlink', href = re.compile("journal")):
			journal_name = journals.get_text()
			not_a_journal = re.compile('Journal List') #exclude this unwanted text
			discard = not_a_journal.search(journal_name)
			if not discard:
				academic_journals.append(journal_name)

		#Get main text
		for words in soup.findAll('p', id = re.compile("__p"), class_ = re.compile('first')):
			document = words.get_text()
			main_text.append(document)
		main_text = ' '.join(main_text)
		print(main_text)

		#print to .txt file
		#sys.stdout = open(str(i)+'.txt', "w")   ## change "w" to "r+" if file exists
		#print (main_text)

		i += 1
		time.sleep(5) #Throttle for 5 seconds
	#return academic_journals


journals_list = get_text(url_list)
#get_text(url_dev)


#main_info = list(zip(titles_list, authors_list, url_list)) #inefficient :'( #list of tuples
#print(len(main_info)) #if this doesn't match url_count, the bot has been blocked on some pages. #rerun with proxy


############ GRAVEYARD #############
#keywords = []
		# for abstracts in soup.findAll('span', {'class': 'kwd-text'}): #keywords #will use as feature
		# 	keyws = abstracts.get_text()
		# 	keywords.append(keyws)
		# 	print(keywords)
# I can't recall where I'm getting these from on a page, and they don't seem to match for the papers...
