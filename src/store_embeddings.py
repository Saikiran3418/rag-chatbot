# import libraries
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

# load API key
load_dotenv()

# Step 1 - load PDF
print("Loading PDF...")
loader = PyMuPDFLoader(r"C:\Users\saiki\rag-chatbot\data\document.pdf")
pages = loader.load()
print(f"✅ PDF loaded — {len(pages)} pages")

# Step 2 - split into chunks
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(pages)
print(f"✅ {len(chunks)} chunks created")

# Step 3 - create embeddings and store in ChromaDB
print("\nStoring chunks in ChromaDB...")
print("(This may take 1-2 minutes, OpenAI is converting text to vectors...)")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="data/chroma_db"  # saved here permanently
)

print(f"✅ All chunks stored in ChromaDB!")
print(f"📁 Database saved at: data/chroma_db")
print(f"\n--- Testing search ---")

# Step 4 - test it with a search
query = "What is the attention mechanism?"
results = vectorstore.similarity_search(query, k=3)

print(f"\n🔍 Question: {query}")
print(f"\nTop 3 most relevant chunks found:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)