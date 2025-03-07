from setuptools import setup, find_packages

setup(
    name="AI Medical ChatBot",
    version="0.0.1",
    author="Aditya Rana",
    author_email="adityaranarana2006@gmail.com",
    packages=find_packages(),
    install_requires=[
        "flask",
        "click",
        "requests",
        "langchain",
        "langchain_pinecone",
        "pinecone-client",
        "langchain_community",
        "python-dotenv",
    ],
)
