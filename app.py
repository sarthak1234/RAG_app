from pathlib import Path
import tempfile

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
import tiktoken

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')

st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation app")

def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf',loader_cls = PyPDFLoader)
    
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def get_document_chunks(documents):
    encoding  = tiktoken.encoding_for_model('text-embedding-ada-002')
    sum = 0
    document_chunks = []
    all_chunks = []
    MAX_TPM = 150000
    MAX_TPM_checker = 0
    for document in documents : 
        encoded_text = encoding.encode(document.page_content)
        sum = sum+len(encoded_text)
        if sum < MAX_TPM : 
            document_chunks.append(document)
        else : 
            sum = 0
            MAX_TPM_checker = 1
            all_chunks.append(document_chunks)
            document_chunks = []
    if MAX_TPM_checker==0 :
        all_chunks.append(document_chunks)
    return all_chunks

def get_faiss_retriever(texts):
    text_chunks = get_document_chunks(texts)
    embeddings = OpenAIEmbeddings()

    for idx,chunk in enumerate(text_chunks):
        if idx==0 : 
            vector_index  = FAISS.from_documents(chunk, embeddings)
        else : 
            vector_index_i = FAISS.from_documents(chunk, embeddings)
            vector_index.merge_from(vector_index_i) 
    retriever = vector_index.as_retriever()
        
    return retriever

def get_response(question,retriever,llm):
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")
    
    
    setup_and_retrieval = RunnableParallel({"context":retriever,"input":RunnablePassthrough()})
    output_parser = StrOutputParser()
    chain = setup_and_retrieval | prompt | llm | output_parser
    response = chain.invoke(question)
    return response

def input_fields():
    
    with st.sidebar:
        
        st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")
    
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)


def process_documents():
    if not st.session_state.openai_api_key  or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                    #
                    with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                        tmp_file.write(source_doc.read())
                    #
                    documents = load_documents()
                    #
                    for _file in TMP_DIR.iterdir():
                        temp_file = TMP_DIR.joinpath(_file)
                        temp_file.unlink()
                    #
                    texts = split_documents(documents)
                    #
                    st.session_state.retriever = get_faiss_retriever(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")



def run_app():
    
    input_fields()
    llm = ChatOpenAI(openai_api_key = st.session_state.openai_api_key)
    st.button("Submit Documents", on_click=process_documents)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = get_response(query,st.session_state.retriever,llm)
        st.chat_message("ai").write(response)

    

if __name__ == '__main__':
    #
    run_app()