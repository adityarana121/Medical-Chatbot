from flask import Flask, request, render_template, jsonify
from src.helper import download_hugging_face_embeddings, llm
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

# ✅ Initialize Flask app
app = Flask(__name__, static_folder="static")

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

# ✅ Debugging: Hide API keys from logs
print("✅ PINECONE API Key Loaded:", "✔️" if api_key else "❌ Missing")
print("✅ GROQ API Key Loaded:", "✔️" if groq_key else "❌ Missing")

# ✅ Ensure API keys exist
if not api_key or not groq_key:
    raise ValueError("❌ Missing API keys. Check your .env file.")

# ✅ Initialize embeddings
embeddings = download_hugging_face_embeddings()

# ✅ Initialize Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key=api_key)
print("📝 Available Pinecone Indexes:", pc.list_indexes().names())

index_name = "test"
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an AI assistant specialized in medical inquiries.\n\n"
        "Context: {context}\n"
        "User Question: {question}\n"
        "If the context does not contain relevant information, say: 'I don't know.'"
    )
)


# ✅ Updated Retrieval-Augmented Generation (RAG) Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"error": "No input message received"}), 400
    
    # ✅ Retrieve relevant documents
    response = qa_chain({"query": msg})
    retrieved_docs = response.get("source_documents", [])

    # ✅ Debugging: Print retrieved documents
    print("🔍 Retrieved Documents:")
    for doc in retrieved_docs:
        print(f"- {doc.page_content[:300]}...")  # Print first 300 characters

    print("🤖 LLM Response:", response.get("result", "No response."))

    return jsonify({"answer": response.get("result", "No response generated.")})
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
