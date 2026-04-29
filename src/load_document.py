# Step 1 - import the libraries we need
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
import os

# Step 2 - load our secret API key from .env file
load_dotenv()

# Step 3 - tell Python where your PDF is
pdf_path = r"C:\Users\saiki\rag-chatbot\data\document.pdf"

# Step 4 - load the PDF
print("Loading PDF...")
loader = PyMuPDFLoader(pdf_path)
pages = loader.load()

# Step 5 - show results
print(f"✅ PDF loaded successfully!")
print(f"📄 Total pages found: {len(pages)}")
print(f"\n--- First 500 characters of Page 1 ---")
print(pages[0].page_content[:500])