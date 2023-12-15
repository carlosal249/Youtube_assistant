import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


import os

os.environ['OPENAI_API_KEY'] = '' # Insert your key here

embeddings = OpenAIEmbeddings()

video_url = 'https://www.youtube.com/watch?v=DJnH0jR8y5Q'

def create_vector_from_yt( video_url:str ) -> FAISS:
    '''
    Creates and store a vector of transcript yt video with splitted format for llm input
    '''

    loader = YoutubeLoader.from_youtube_url(video_url)
    trasncript = loader.load()
    
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = text_spliter.split_documents(trasncript)
    db = FAISS.from_documents(docs, embeddings)

    return db

def get_response_from_query(db, query, k=4) -> str:
    # text-davincii - 4097 tokens
    # k = docs/tokens

    docs = db.similarity_search(query, k=k)
    docs_page_content = ' '.join([d.page_content for d in docs])

    llm = OpenAI(model='text-davinci-003')
    prompt = PromptTemplate(
        input_variables=['question', 'docs'],
        template= """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")

    return response, docs

#print(create_vector_from_yt(video_url))