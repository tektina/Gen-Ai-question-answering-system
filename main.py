import streamlit as st
from helper import load_and_store
from dotenv import load_dotenv
import os

load_dotenv()

st.title("📄 AI Document Q&A System")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("data/temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    qa = load_and_store("data/temp.pdf")

    question = st.text_input("Ask a question from the document")

    if question:
        answer = qa.run(question)
        st.success(answer)