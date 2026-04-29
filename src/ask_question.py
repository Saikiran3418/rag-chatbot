# import libraries
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# load API key
load_dotenv()

# Step 1 - connect to your existing ChromaDB
print("Loading vector database...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Vector database loaded!")

# Step 2 - create ChatGPT
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# Step 3 - create prompt template
prompt = PromptTemplate.from_template("""
You are a helpful assistant that answers questions 
about the provided document.

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

# Step 4 - helper function to format chunks
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 5 - build the pipeline
chain = (
    {"context": retriever | format_docs, 
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 6 - chat loop
print("\n" + "="*50)
print("🤖 RAG Chatbot Ready!")
print("Type your question and press Enter")
print("Type 'quit' to exit")
print("="*50 + "\n")

while True:
    question = input("You: ")

    if question.lower() == "quit":
        print("Goodbye!")
        break

    if question.strip() == "":
        continue

    print("\nThinking...")
    answer = chain.invoke(question)
    print(f"\n🤖 Answer: {answer}")
    print("\n" + "-"*50 + "\n")