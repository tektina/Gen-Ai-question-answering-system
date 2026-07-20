from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI

def load_and_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=db.as_retriever()
    )
    return qa