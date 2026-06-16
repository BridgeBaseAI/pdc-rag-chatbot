import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = "data_sources"
CHROMA_PATH = "chroma"

print("Loading PDFs...")
documents = []
for filename in os.listdir(DATA_PATH):
    if filename.endswith(".pdf"):
        filepath = os.path.join(DATA_PATH, filename)
        print(f"  Loading: {filename}")
        loader = PyPDFLoader(filepath)
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages total")

print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Saving to vector database...")
db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
print("Done! Vector database created.")