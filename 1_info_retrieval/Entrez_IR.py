from Bio import Entrez
import time, sys
from time import sleep
import xml.etree.ElementTree as ET

#Entrez Information Retrieval
#This code uses BioPython to access NCBI's API (Entrez)
#From the NCBI API, this code references PubMed and PubMedCentral
#With an input PubMed ID's (pmids), the code will retrieve information about the original pmid,
#and information about pubplications that cite this pmid via PubMedCentral ID's (pmcids)


Entrez.email = "hclent1@gmail.com"
Entrez.tool = "MyInfoRetrieval"

#my_pmid = "18952863"
#How to usefully compare homologous plant genes and chromosomes as DNA sequences.
#pmid = 18269575

#Finding and Comparing Syntenic Regions among Arabidopsis
my_pmid = 19110700


#Input: pmid
#Output: basic info on pmid
def getMainInfo(pmid):
	handle = Entrez.esummary(db="pubmed", id=pmid)
	record = Entrez.read(handle)
	title = record[0]["Title"]
	authors = record[0]["AuthorList"]
	journal = record[0]["FullJournalName"]
	return title, authors, journal


#Input: Pmid
#Output: list of pmcids of the articles citing you
def getCitationIDs(pmid): #about the same speed as MainCrawl.py
	results = Entrez.read(Entrez.elink(dbfrom="pubmed", db="pmc", LinkName="pubmed_pmc_refs", from_uid=pmid))
	pmc_ids = [link["Id"] for link in results[0]["LinkSetDb"][0]["Link"]]
	return pmc_ids
	time.sleep(3)


#Input: Citing pmcids
#Output: Basic info about these pmcids
def getCitedInfo(pmcid_list): 
	pmc_titles = []
	pmc_authors = []
	pmc_journals = []
	pmc_urls = []

	for citation in pmcid_list:
		handle = Entrez.esummary(db="pmc", id=citation)
		record = Entrez.read(handle)
		t = record[0]["Title"]
		a = record[0]["AuthorList"]
		j = record[0]["FullJournalName"]
		u = "http://www.ncbi.nlm.nih.gov/pmc/articles/PMC"+citation
		pmc_titles.append(t)
		pmc_authors.append(a)
		pmc_journals.append(j)
		pmc_urls.append(u)
		time.sleep(3)
	return pmc_titles, pmc_authors, pmc_journals, pmc_urls


#Input: XML string of PMC entry generated with getContentPMC
#Output: Abstract and journal text 
#Very rarely Entrez is unable to provide the XML or there is no main text. Parse PMC handles this.
def parsePMC(xml_string, pmid):
	main_text = []
	root = ET.fromstring(xml_string) #parsed
	#Get abstract and add to doc
	try:
		abstract = root.find('.//abstract')
		full_abs = ("".join(abstract.itertext()))
		#print("* Got abstract")
		main_text.append(full_abs)
	except Exception as e:   
		print("The following PMCID is not available") 
	try:
		#Get main text and add to doc
		text = root.findall('.//p')
		for t in text:
			full_text = ("".join(t.itertext()))
			main_text.append(full_text)
		#print("* Got main text")
	except Exception as e:
		print("Only gave us the absract")
	return main_text



#Input: pmid and the list of pmcids citing it
#For each citing pmc_id, this function gest the xml, which is then parsed by parsePMC()
#Output: Journal texts for each pmcid
def getContentPMC(pmid, pmcids_list):
	i = 1
	for citation in pmcids_list:
		handle = Entrez.efetch(db="pmc", id=citation, rettype='full', retmode="xml")
		xml_record = handle.read() #xml str
		main_text = parsePMC(xml_record, pmid)
		sys.stdout = open(str(pmid)+'_'+str(i)+'.txt', "w")
		print(main_text)
		i += 1
		time.sleep(3)



pmc_ids = getCitationIDs(my_pmid)
print("CITED PMC IDS: ")
print(pmc_ids)
amount = len(pmc_ids)
print(amount)
print("THERE ARE " + str(amount) + " DOCUMENTS")
getContentPMC(my_pmid, pmc_ids)


