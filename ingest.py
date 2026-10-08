from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Load the PDF
loader = PyPDFLoader("documents/jawa.pdf")
documents = loader.load()

print("Number of pages:", len(documents))


# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# 3. Create embeddings
print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Store embeddings in FAISS
print("Creating FAISS vector store...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)


# 5. Save FAISS locally
vectorstore.save_local("faiss_index")

print("FAISS vector store created successfully!")