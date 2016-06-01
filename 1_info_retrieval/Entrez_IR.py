from Bio import Entrez
import time, sys
from time import sleep
import xml.etree.ElementTree as ET

#Entrez Information Retrieval
#This code uses BioPython to access NCBI's API (Entrez)
#From the NCBI API, this code references PubMed and PubMedCentral
#With an input PubMed ID's (pmids), the code will retrieve information about the original pmid,
#and information about pubplications that cite this pmid via PubMedCentral ID's (pmcids)


Entrez.email = "xxxxx@xxxxxxx.xxx"
Entrez.tool = "MyInfoRetrieval"

my_pmid = "18269575"


#Input: pmid
#Output: basic info on pmid
def getMainInfo(pmid):
	handle = Entrez.esummary(db="pubmed", id=pmid)
	record = Entrez.read(handle)
	title = record[0]["Title"]
	authors = record[0]["AuthorList"]
	journal = record[0]["FullJournalName"]
	print(title)
	print(authors)
	print(journal)
	print("* * * * * * * * * * * * * * * ")


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
	for citation in pmcid_list:
		handle = Entrez.esummary(db="pmc", id=citation)
		record = Entrez.read(handle)
		title = record[0]["Title"]
		authors = record[0]["AuthorList"]
		journal = record[0]["FullJournalName"]
		url = "http://www.ncbi.nlm.nih.gov/pmc/articles/PMC"+citation
		print(title)
		print(authors)
		print(journal)
		print(url)
		print("* * * * * * * * * * * * * * * ")
		time.sleep(3)


#Input: XML string of PMC entry generated with getContentPMC
#Output: Abstract and journal text 
def parsePMC(xml_string, pmid):
	main_text = []
	root = ET.fromstring(xml_string) #parsed
	#Get abstract and add to doc
	abstract = root.find('.//abstract')
	full_abs = ("".join(abstract.itertext()))
	main_text.append(full_abs)
	#Get main text and add to doc
	text = root.findall('.//p')
	for t in text:
		full_text = ("".join(t.itertext()))
		main_text.append(full_text)
	return main_text


#Input: pmid and the list of pmcids citing it
#For each citing pmc_id, this function gest the xml, which is then parsed by parsePMC()
#Output: Journal texts for each pmcid
def getContentPMC(pmid, pmcids_list):
	i = 4
	for citation in pmcids_list:
		handle = Entrez.efetch(db="pmc", id=citation, rettype='full', retmode="xml")
		xml_record = handle.read() #xml str
		main_text = parsePMC(xml_record, pmid)
		sys.stdout = open(str(pmid)+'_'+str(i)+'.txt', "w")
		print(main_text)
		i += 1
		time.sleep(3)


getMainInfo(my_pmid)
pmc_ids = getCitationIDs(my_pmid)
#print("CITED PMC IDS: ")
#print(pmc_ids)
#amount = len(pmc_ids)
#print("THERE ARE " + str(amount) + " DOCUMENTS")
getCitedInfo(pmc_ids)
getContentPMC(my_pmid, pmc_ids)


