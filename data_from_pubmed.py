#!/usr/bin/env python
# coding: utf-8

#Importing biopython and pandas packages
from Bio import Entrez
import pandas as pd


#Add your email id to extract results from pubmed database. Change parameters accordingly.
#Gets the search results from pubmed based on the search terms
def search(query):
    Entrez.email = 'Enter your email id here'
    handle = Entrez.esearch(db='pubmed', 
                            sort='relevance', 
                            retmax='200',
                            retmode='xml', 
                            term=query)
    results = Entrez.read(handle)
    return results


#Extract the id's of the papers  
def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'Enter your email id here'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           rettype='abstract',
                           id=ids)
    results = Entrez.read(handle)
    return results

#eg search term: opioid AND nicotine
results = search("") #USE AND OR and NOT for boolean logic
id_list = results['IdList']

#print(results)
print(id_list)
print(len(id_list))
print(results)

papers= fetch_details(id_list)
print(papers)

article_list=list()
#Create a list of paper abstracts and titles
for record in papers["PubmedArticle"]:
    #print(record['MedlineCitation']['Article'])
    keys=record['MedlineCitation']['Article'].keys()
    #print(type(keys))
    if("Abstract" in keys):
        
        article_title=record["MedlineCitation"]["Article"]["ArticleTitle"]
        #print(type(article_title))
        article_abstract=record['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
        #print(article_abstract)
        article_list.append([article_title,article_abstract])
print(article_list)

#Save the result to a csv file

df = pd.DataFrame(article_list, columns = ['Title', 'Abstract'])
df.to_csv("Enter csv filename here",index=False, header=True)





