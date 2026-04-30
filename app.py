# import libraries
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load API key
load_dotenv()
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# page settings
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🤖",
    layout="centered"
)

# title
st.title("🤖 RAG Document Chatbot")
st.caption("Upload any PDF and ask questions about it!")
st.divider()

# ---- SIDEBAR ----
with st.sidebar:
    st.header("📄 Upload Your Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf"
    )

    if uploaded_file is not None:
        st.success(f"✅ {uploaded_file.name} uploaded!")
        process_btn = st.button(
            "🔄 Process Document",
            type="primary",
            use_container_width=True
        )
    else:
        st.info("👆 Upload a PDF to get started")
        process_btn = False

    st.divider()
    st.markdown("**How to use:**")
    st.markdown("1. Upload a PDF file")
    st.markdown("2. Click Process Document")
    st.markdown("3. Ask questions below!")

# ---- PROCESS UPLOADED PDF ----
def process_pdf(uploaded_file):
    # save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # load and split
    loader = PyMuPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(pages)

    # create IN-MEMORY vector store (works on cloud!)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # no persist_directory = stays in memory only
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # clean up temp file
    os.unlink(tmp_path)

    return vectorstore

# ---- BUILD QA CHAIN ----
def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    prompt = PromptTemplate.from_template("""
You are a helpful assistant that answers questions 
about the uploaded document.

Rules:
1. Answer ONLY from the context below
2. If the answer is not in context, say:
   "I don't have enough information to answer this."
3. Keep answers clear and simple

Context:
{context}

Question:
{question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ---- PROCESS BUTTON CLICKED ----
if process_btn and uploaded_file is not None:
    # clear everything from previous document
    if "chain" in st.session_state:
        del st.session_state.chain
    if "messages" in st.session_state:
        del st.session_state.messages
    if "doc_name" in st.session_state:
        del st.session_state.doc_name

    with st.spinner("Processing your PDF... please wait!"):
        vectorstore = process_pdf(uploaded_file)
        st.session_state.chain = build_chain(vectorstore)
        st.session_state.messages = []
        st.session_state.doc_name = uploaded_file.name
    st.success("✅ Document processed! Ask your questions!")

# ---- CHAT INTERFACE ----
if "chain" not in st.session_state:
    st.info(
        "👈 Upload a PDF in the sidebar to get started!"
    )
else:
    st.markdown(
        f"**Chatting about:** {st.session_state.doc_name}"
    )
    st.divider()

    # show previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat input
    if question := st.chat_input(
        "Ask a question about your document..."
    ):
        # show user question
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        # get and show answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(
                    question
                )
                st.markdown(answer)

        # save to history
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )