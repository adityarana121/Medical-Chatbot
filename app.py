from flask import Flask, request, render_template,jsonify
from src.helper import download_hugging_face_embeddings,llm
from store_index import text_chunks
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)

load_dotenv()

api_key=os.environ.get("PINECONE_API_KEY")
groq_key=os.environ.get("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = api_key
os.environ["GROQ_API_KEY"] = "groq_key"

embeddings=download_hugging_face_embeddings()

index_name = "newtest"

# embedded each chunk and upsert the embeddingd into the Pinecone Index
from langchain_pinecone import PineconeVectorStore
docsearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human","{input}"),
    ]
)

question_answer_chain=create_stuff_documents_chain(llm, prompt)
rag_chain=create_retrieval_chain(retriever, question_answer_chain)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/get',methods=['GET','POST'])
def chat():
    msg=request.form("msg")
    input=msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("response",response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0",port= 8080,debug=True)