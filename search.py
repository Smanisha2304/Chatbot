from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Load the FAISS vector store
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# 3. Create a retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# 4. Ask a question
question = input("Ask a question: ")


# 5. Retrieve relevant chunks
results = retriever.invoke(question)


# 6. Display retrieved information
print("\nRelevant information:\n")

for i, document in enumerate(results):
    print(f"--- Result {i + 1} ---")
    print(document.page_content)
    print()