# import libraries
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load API key
load_dotenv()

# load the PDF
print("Loading PDF...")
loader = PyMuPDFLoader(r"C:\Users\saiki\rag-chatbot\data\document.pdf")
pages = loader.load()
print(f"✅ PDF loaded — {len(pages)} pages found")

# split into chunks
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # each chunk = max 500 characters
    chunk_overlap=50,     # 50 characters overlap between chunks
)

chunks = splitter.split_documents(pages)

# show results
print(f"✅ Splitting done!")
print(f"📦 Total chunks created: {len(chunks)}")
print(f"\n--- Example: Chunk number 1 ---")
print(chunks[0].page_content)
print(f"\n--- Example: Chunk number 2 ---")
print(chunks[1].page_content)
print(f"\n--- Example: Chunk number 3 ---")
print(chunks[2].page_content)