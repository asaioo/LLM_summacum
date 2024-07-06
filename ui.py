__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import tempfile
import array

import os
from dotenv import load_dotenv


import oracledb
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import BaseDocumentTransformer, Document

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import UpstageEmbeddings, UpstageLayoutAnalysisLoader
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

import warnings
warnings.filterwarnings("ignore")



username=os.environ["DB_USER"]
password=os.environ["DB_PASSWORD"]
dsn=os.environ["DSN"]

con = oracledb.connect(user=username, password=password, dsn=dsn)

try: 
    conn23c = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!", conn23c.version)
except Exception as e:
    print("Connection failed!")

st.title("Summacum")
st.write("---")

uploaded_file = st.file_uploader("Choose a file to study",type=['pdf'])
st.write("---")

def preprocess(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader(temp_filepath).load()

    file_path = temp_filepath

    layzer = UpstageLayoutAnalysisLoader(file_path,split="page")
    text_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=1500, chunk_overlap=200, language=Language.HTML
)
    docs = text_splitter.split_documents(docs)
    '''for doc in docs:
        doc.metadata['title']="Biomedical Engineering_ Bridging Medicine and Technology"
    '''
    
    return docs

def embedding(docs):
    upstage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
     
# Configure the vector store with the model, table name, and using the indicated distance strategy for the similarity search and vectorize the chunks

    knowledge_base = OracleVS.from_documents(docs, upstage_embeddings, client=conn23c, 
                    table_name="biomedical_table", 
                    distance_strategy=DistanceStrategy.DOT_PRODUCT)
    
    upstage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

    vector_store = OracleVS(client=conn23c, 
                        embedding_function=upstage_embeddings, 
                        table_name="biomedical_table", 
                        distance_strategy=DistanceStrategy.DOT_PRODUCT)

    retriever = vector_store.as_retriever()
    
    return retriever, vector_store

def get_data():
    uploaded_problem = st.file_uploader("give a problem",type=['pdf'])
    uploaded_answer = st.file_uploader("give an answer",type=['pdf'])
    prob_docs = preprocess(uploaded_problem)
    prob_sols = preprocess(uploaded_answer)
    
    return prob_docs, prob_sols

def main():
    if uploaded_file is not None:
        docs = preprocess(uploaded_file) 
        retriever, vector_store = embedding(docs)
        prb_docs, sol_docs = get_data()
        from langchain_upstage import UpstageGroundednessCheck

        groundedness_check = UpstageGroundednessCheck()
        llm = ChatUpstage()
        template = """Please give me the solutions from quiz referencing the below knowledge:
              ----------
              {context}
              ----------
              Quiz: {quiz}
              """
        prompt = PromptTemplate.from_template(template)
        k=4
        grounded = False
        while not grounded:
            retriever = vector_store.as_retriever(k=k)

            chain = (
    # {"context": retriever, "quiz": RunnablePassthrough(), "sol": RunnablePassthrough()}
                {"context": retriever, "quiz": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
                    )
  # response = chain.invoke({"quiz":prb_docs[0], "sol":sol_docs[0]})
            response = chain.invoke({"quiz":prb_docs[0].page_content})
            st.write("Bot response: ", response)

  
            result_chunks=vector_store.similarity_search(prb_docs[0].page_content, k=k)
            ref = ""
            for i in range(k):
                ref += result_chunks[i].page_content
                ref += "\n"
                gc_result = groundedness_check.invoke({"context": ref, "answer": response})

            if gc_result.lower().startswith("grounded"):
                grounded = True
                st.write(f"Answer is grounded\n[ref] {ref}")
            else:
                st.write(f"answer is not grounded with k={k}")
                k *=2
        

        
           
    
if __name__ == "__main__":
    main()
    