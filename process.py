"""
export GOOGLE_API_KEY="replace with your key here"
export USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

import langchain
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.chains import create_retrieval_chain
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import html2text
import time
import logging
import json

llm = ChatGoogleGenerativeAI(model="gemini-pro")
langchain.debug = True
articles_directory = 'fx-streets'

print("\033[1;32m Step 1: Fetching articles listing from https://www.fxstreet.com/news using python BeautifulSoup and selenium web driver")

def fetchListing():
    driver = webdriver.Chrome()
    driver.get("https://www.fxstreet.com/news")
    timeout = 30  

    WebDriverWait(driver, timeout).until(
        expected_conditions.presence_of_element_located(            
             (By.CSS_SELECTOR, "div.ais-hits")
        )
    )
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    div = soup.find('div', class_='ais-hits')
    urls = [] 
    selected_div = soup.select_one('div.ais-hits')
    for link in selected_div.select('div.ais-hits--item h4.fxs_headline_tiny a[href]'):
        urls.append(link['href'])
    return urls

listing_urls = fetchListing()
print(json.dumps(listing_urls, indent=4))

print("\033[1;32m Step 2: Fetching individual article details using Urls List, Transform html into raw text, Then save into directory")

def fetchTransformStoreArticle(link,articles_directory):
    try:
        url = [link]
        loader = AsyncHtmlLoader(url)
        docs = loader.load()

        html2text_transformer = Html2TextTransformer()
        docs_transformed = html2text_transformer.transform_documents(docs)

        if docs_transformed != None and len(docs_transformed) > 0:
            metadata = docs_transformed[0].metadata
            title = metadata.get('title', '')
            arr = link.split('/news/')
            article = arr[1]
            file = open(articles_directory+'/'+article+'.txt', 'w')
            file.write(html2text.html2text(docs_transformed[0].page_content))
            file.close() 
            return title+" article saved successfully"                
        else:
            return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

listing_urls = listing_urls[:10]

for link in listing_urls:
      response = fetchTransformStoreArticle(link,articles_directory)
      if response != None:
        print(response)

print("\033[1;32m Step 3: Segment documents into chunks and apply vector embedding for subsequent utilization by ChromaDB")

def loadDocuments(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = loadDocuments(articles_directory)
#print(len(documents))

def segmentDocuments(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = segmentDocuments(documents)
#print(len(docs))

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#db = Chroma.from_documents(docs, embeddings)
persist_directory = "chroma_db"
vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
vectordb.persist()
#vectordb = None

print("\033[1;32m Step 4: Retrieve relevant documents from vector db for further processing in Langchain RAG pipeline")

retriever = vectordb.as_retriever(search_type="mmr",search_kwargs={"k": 10})
input = "EUR & USD"
template = """
You are expert in forex market analysis, Act as a helpful AI assistant.
Answer based on the context provided. 
context: {context}
input: {input}
summarize each document including sentiment analysis in plain english against each forex pair provided in input
format the output in json format for each document with respected keys ie pair,sentiment,summary
"""

prompt = PromptTemplate.from_template(template=template)
#print('prompt',prompt)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
#response=retrieval_chain.invoke({"input":"GBP/USD EUR/GBP"})
response=retrieval_chain.invoke({"input":input})
print(response["answer"])

print("\033[1;32m Step 5: Extracting key concepts & terminologies used in forex articles")

docs_list = []
docs = retriever.invoke(input)
for doc in docs:
    docs_list.append(doc.page_content)

first_prompt = ChatPromptTemplate.from_template(
    "you are forex expert, summarize & extract top 5 keywords as key concepts used in financial, trading & investing domain {data} ?"
)
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="concepts")

second_prompt = ChatPromptTemplate.from_template(   
    "Explain each concept in plain english with 1 line sentence? {concepts}, Answer the output as comma separated Python list for each concept in python dict format"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],verbose=True)
response = overall_simple_chain.invoke(docs_list)
print(response["output"])
