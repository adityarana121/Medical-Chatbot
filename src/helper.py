import os
from dotenv import load_dotenv

# ✅ Load .env file
dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path)

# ✅ Debugging: Check API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY is missing. Check your .env file.")

# ✅ Correct Import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# ✅ Functions
def load_pdf_file(data):
    return DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader).load()

def split_text(extracted_data):
    return RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20).split_documents(extracted_data)

def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

llm = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768", temperature=0.5, max_tokens=400)

# ✅ Debugging: Check LLM response
print("Testing LLM:", llm.invoke("Hello"))
