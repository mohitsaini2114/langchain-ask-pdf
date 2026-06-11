from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage


def main():
    st.set_page_config(page_title="Ask your PDF")
    load_dotenv()
    print("API KEY:", os.getenv("ANTHROPIC_API_KEY"))  # remove after testing
    st.header("Ask your PDF👨🏻‍💻")
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
            prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {user_question}"
            response = llm.invoke([HumanMessage(content=prompt)])
            st.write(response.content)


if __name__ == '__main__':
    main()