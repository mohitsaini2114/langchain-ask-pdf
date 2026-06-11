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
    st.set_page_config(page_title="Ask your PDFs")
    load_dotenv()

    st.header("Ask your PDFs👨🏻‍💻")

    # init session state
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None
    if "uploaded_pdfs" not in st.session_state:
        st.session_state.uploaded_pdfs = []

    # sidebar
    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdfs = st.file_uploader(
            "Upload one or more PDFs",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        if pdfs:
            for pdf in pdfs:
                names = [p.name for p in st.session_state.uploaded_pdfs]
                if pdf.name not in names:
                    st.session_state.uploaded_pdfs.append(pdf)

        if st.session_state.uploaded_pdfs:
            st.write(f"📄 {len(st.session_state.uploaded_pdfs)} PDF(s) loaded:")
            for p in st.session_state.uploaded_pdfs:
                st.write(f"- {p.name}")

        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button("Process PDFs")
        with col2:
            if st.button("Clear All"):
                st.session_state.uploaded_pdfs = []
                st.session_state.knowledge_base = None
                st.rerun()

    if process_btn and st.session_state.uploaded_pdfs:
        with st.spinner("Processing PDFs..."):
            text = ""
            for pdf in st.session_state.uploaded_pdfs:
                pdf.seek(0)  # reset file pointer
                pdf_reader = PdfReader(pdf)
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
            st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)
            st.success(f"✅ {len(st.session_state.uploaded_pdfs)} PDF(s) processed!")

    elif process_btn:
        st.warning("Please upload at least one PDF first.")

    if st.session_state.knowledge_base:
        user_question = st.text_input("Ask a question about your PDFs:")
        if user_question:
            with st.spinner("Thinking..."):
                docs = st.session_state.knowledge_base.similarity_search(user_question)
                context = "\n\n".join([doc.page_content for doc in docs])

                llm = ChatAnthropic(
                    model="claude-haiku-4-5-20251001",
                    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {user_question}"
                response = llm.invoke([HumanMessage(content=prompt)])
                st.write(response.content)


if __name__ == '__main__':
    main()