from src.helper import load_pdf_file, split_text, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

# Extract and process text data
extracted_data = load_pdf_file("data")
text_chunks = split_text(extracted_data)
embeddings = download_hugging_face_embeddings()

# ✅ Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Index name
index_name = "newtest"

# ✅ Check if the index exists before creating it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Ensure this matches your embeddings
        metric="cosine",  # or "dotproduct" / "euclidean"
        spec=ServerlessSpec(
            cloud="aws",  # ✅ Change to "aws" (Pinecone free-tier supports AWS)
            region="us-east-1"  # ✅ Supported free-tier region
        )
    )
else:
    print(f"✅ Index '{index_name}' already exists. Skipping creation.")

# ✅ Connect to the existing Pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
