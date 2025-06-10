import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import tempfile
import os

st.set_page_config(page_title='ClearRoute Sales Chatbot', layout='wide')
st.title("ClearRoute Sales Chatbot")

# Sidebar for API key and file upload
with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    st.markdown("[Get your OpenAI API key](https://platform.openai.com/api-keys)")

if "history" not in st.session_state:
    st.session_state.history = []

# Custom prompt: restrict answers to context only
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant for answering questions about uploaded PDF documents. "
        "Always answer using ONLY the provided context. "
        "If the answer is not in the context, simply reply: "
        "'Out of context. No relevant information found in the uploaded PDFs.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
)

def process_pdfs(files):
    docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            loader = PyPDFLoader(tmp_file.name)
            for doc in loader.load():
                doc.metadata["source"] = file.name  # Store original filename
                docs.append(doc)
            os.unlink(tmp_file.name)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    return splitter.split_documents(docs)

def build_vector_store(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)

if openai_api_key and uploaded_files:
    if "vector_store" not in st.session_state or st.session_state.get("last_files") != uploaded_files:
        with st.spinner("Processing and indexing PDFs..."):
            chunks = process_pdfs(uploaded_files)
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vector_store = build_vector_store(chunks, embeddings)
            st.session_state.vector_store = vector_store
            st.session_state.last_files = uploaded_files

    chat_model = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name="gpt-3.5-turbo"  # Or "gpt-4o" if you have access
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        chat_model,
        retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )

    st.divider()
    st.subheader("Ask any question about your PDFs")
    user_input = st.chat_input("Type your question here...")

    if user_input:
        with st.spinner("Generating answer..."):
            result = qa_chain({"question": user_input, "chat_history": st.session_state.history})
            answer = result["answer"].strip()
            # If the answer is not in context, enforce the out-of-context message
            if "Out of context." in answer or answer.lower().startswith("out of context"):
                answer = "Out of context. No relevant information found in the uploaded PDFs."
                sources = []
            else:
                sources = result.get("source_documents", [])
            # Format sources
            if sources:
                source_names = set(doc.metadata.get("source", "PDF") for doc in sources)
                answer += "\n\n**Sources:** " + ", ".join(source_names)
            st.session_state.history.append((user_input, answer))

    # Display chat history with nice formatting
    for q, a in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

else:
    st.info("Please upload PDF files and enter your OpenAI API key to get started.")
