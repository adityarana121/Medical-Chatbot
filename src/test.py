from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# ✅ Load API keys
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
index_name = "test"

# ✅ Connect to Pinecone
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Test retrieval
query = "What is acne?"
retrieved_docs = retriever.invoke(query)

# ✅ Print retrieved documents
print("🔍 Retrieved Documents:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"Chunk {i}: {doc.page_content[:300]}...")
