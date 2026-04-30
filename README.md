# 🤖 RAG Document Chatbot

An End-to-End Retrieval Augmented Generation (RAG) 
chatbot that answers questions from any PDF document.

## 🚀 Live Demo
👉 https://rag-chatbot-dwbhq5qrbretn6cirg2boe.streamlit.app/

## 📸 How it works
1. Upload any PDF document
2. Ask questions about it
3. Get accurate AI-powered answers with sources

## 🛠️ Tech Stack
- **LangChain** — AI pipeline framework
- **OpenAI GPT-4o-mini** — Language model
- **ChromaDB** — Vector database
- **OpenAI Embeddings** — Text vectorization
- **PyMuPDF** — PDF parsing
- **Streamlit** — Chat interface

## ⚙️ Setup Instructions

### 1. Clone the repo
git clone https://github.com/Saikiran3418/rag-chatbot.git
cd rag-chatbot

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

### 3. Install libraries
pip install -r requirements.txt

### 4. Add your OpenAI API key
Create a `.env` file and add:
OPENAI_API_KEY=your-openai-api-key-here

### 5. Run the app
streamlit run app.py

## 📁 Project Structure
rag-chatbot/
├── app.py              ← Main Streamlit chat interface
├── src/
│   ├── load_document.py    ← PDF loader
│   ├── chunk_document.py   ← Text chunking
│   ├── store_embeddings.py ← Vector database
│   └── ask_question.py     ← QA chain
├── data/               ← PDF storage
├── requirements.txt    ← Dependencies
└── .env               ← API keys (not uploaded)

## 🎯 Features
- Upload any PDF directly in the browser
- Intelligent chunking with overlap
- Semantic search using OpenAI embeddings
- Answers grounded only in your document
- Chat history within session
- Source citations for every answer

## 👨‍💻 Built By
Sai Kiran Reddy
