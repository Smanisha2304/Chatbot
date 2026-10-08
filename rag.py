from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()


# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Load FAISS
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# 3. Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# 4. Create Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# 5. Ask a question
question = input("Ask a question about Jawa motorcycles: ")


# 6. Retrieve relevant information
documents = retriever.invoke(question)


# 7. Combine retrieved information
context = "\n\n".join(
    document.page_content
    for document in documents
)


# 8. Send retrieved information to Groq
prompt = f"""
Answer the question using only the information provided in the context below.

If the answer is not available in the context, say:
"I don't know based on the provided Jawa document."

Context:
{context}

Question:
{question}

Answer:
"""


# 9. Generate final answer
response = llm.invoke(prompt)


print("\nAnswer:")
print(response.content)