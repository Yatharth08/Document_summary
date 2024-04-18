import os
import time
import streamlit as st
import json
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent

import numpy
import numpy.linalg
import numpy.linalg._umath_linalg
import csv

openai.api_key = <open_ai_key>

# Helper funciton to save pdf and word file in report folder
def save_file(file):
  save_path = os.path.join(".", "Report", file.name)
  with open(save_path, "wb") as f:
      f.write(file.getbuffer())
  return save_path

# Helper function to read the pdf file in report folder
def read_pdf_file():
  pdf_loader = DirectoryLoader('./Report', glob='**/*.pdf')
  documents = []
  documents.extend(pdf_loader.load())
  return documents

# Helper function to read the doc file in the report folder
def read_doc_file():
  doc_loader = DirectoryLoader('./Report', glob='**/*.docx')
  documents = []
  documents.extend(doc_loader.load())
  return documents


# Helper function to extract text from pdf and docx file
def extract_text(documents):
  text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)  # Adjust chunk size to reduce tokens
  documents = text_splitter.split_documents(documents)
  return documents

# Helper function to get pdf and docx summary
def get_summary(documents):
  embeddings = OpenAIEmbeddings(openai_api_key = <open_ai_key>)
  vectorstore = Chroma.from_documents(documents, embeddings)
  retrievalQA = RetrievalQA.from_llm(llm=OpenAI(openai_api_key=<open_ai_key>),
                                  retriever=vectorstore.as_retriever())

  text = "Understand the text and give summary in English language."
  english_summary = retrievalQA.run(text)
  prompt = f"""Convert text which is delimited by triple backticks ```{english_summary}``` to french language"""
  llmresponse = get_completion(prompt)
  st.subheader("English summary:")
  st.write(english_summary)
  st.subheader("French summary:")
  st.write(llmresponse)


# Remove files from report folder
def remove_files_from_report_folder():
  folder_path = "./Report"
  for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)

def get_completion(prompt, model="gpt-3.5-turbo"):
  messages = [{"role": "user", "content": prompt}]
  response = openai.chat.completions.create(
      model=model,
      messages=messages,
      temperature=0
  )
  print(response)
  return response.choices[0].message.content

def main():
  st.subheader("Document Sensei: Simplify Your Reading Experience")
  menu = ["PDF", "DocumentFiles", "Dataset"]
  choice = st.sidebar.selectbox("Select document type",menu)

  # if csv file is uploaded
  if  choice == "Dataset":
    st.subheader("Dataset")
    data_file = st.file_uploader("Upload CSV",type=['csv'])
    if st.button("Summary"):
      if data_file is not None:
        llm = OpenAI(openai_api_key=<open_ai_key>)
        agent = create_csv_agent(llm, data_file)
        english_summary = agent.run("Understand the dataset and give the summary of what the data is about in English language.")
        prompt = f"""Convert text which is delimited by triple backticks ```{english_summary}``` to french language"""
        llmresponse = get_completion(prompt)
        st.write(english_summary)
        st.write(llmresponse)

  # if document file is uploaded
  if  choice == 'DocumentFiles':
    st.subheader("DocumentFiles")
    doc_file = st.file_uploader("Upload File",type=['docx'])
    if st.button("Summary"):
      if doc_file is not None:
        # Save file
        save_file(doc_file)

        # Read file that is in report folder
        documents = read_doc_file()

        # Extract text from doc
        documents = extract_text(documents)

        # Give summary
        get_summary(documents)

        # Remove file from report folder
        remove_files_from_report_folder()

  # if pdf file is uploaded
  if  choice == "PDF":
    st.subheader("PDF")
    pdf_file = st.file_uploader("Upload File",type=['pdf'])

    if st.button("Summary"):
      if pdf_file is not None:

        # Save file
        save_file(pdf_file)

        # Read file that is in report folder
        documents = read_pdf_file()

        # Extract text from pdf
        documents = extract_text(documents)

        # Give summary
        get_summary(documents)

        # Remove file from report folder
        remove_files_from_report_folder()

main()


