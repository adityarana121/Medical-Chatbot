import os
from dotenv import load_dotenv

# ✅ Load .env file
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path)

# ✅ Ensure API keys exist
if not os.getenv("GROQ_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    raise ValueError("❌ API keys missing. Check your .env file.")

# ✅ Import modules AFTER loading .env
from src.helper import load_pdf_file, split_text, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# ✅ Initialize Pinecone connection
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

# ✅ Debugging: Check Pinecone connection
print("📝 Available Pinecone Indexes:", pc.list_indexes().names())

# ✅ Define Pinecone index
index_name = "test"

# ✅ Check if the index exists before creating it
if index_name not in pc.list_indexes().names():
    print("✅ Creating new Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  
    )
else:
    print(f"✅ Index '{index_name}' already exists.")

# ✅ Connect to Pinecone index
index = pc.Index(index_name)

# ✅ Get stored vector count
vector_count = index.describe_index_stats()["total_vector_count"]
print(f"📊 Pinecone currently stores {vector_count} chunks.")

# ✅ Extract and process text data
print("📂 Loading PDF files from 'data/' directory...")
if not os.path.exists("data"):
    raise FileNotFoundError("❌ 'data/' directory not found! Please add PDFs.")

extracted_data = load_pdf_file("data")
if not extracted_data:
    raise ValueError("❌ No PDFs found in 'data/'. Please add files.")

# ✅ Splitting text
text_chunks = split_text(extracted_data)
print("📄 First 5 chunks to be indexed:")
for i, chunk in enumerate(text_chunks[:5]):
    print(f"Chunk {i+1}: {chunk.page_content[:300]}...")

# ✅ Indexing only if new data is present
if vector_count < len(text_chunks):
    print("✅ Indexing new chunks...")
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=download_hugging_face_embeddings(),
    )
    print(f"✅ Successfully indexed {len(text_chunks)} chunks!")
