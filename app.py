import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Jawa AI",
    page_icon="🏍️",
    layout="centered"
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #f5f5f5;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 25px 0 10px 0;
    }

    .main-header h1 {
        font-size: 38px;
        margin-bottom: 5px;
        color: #111111;
    }

    .main-header p {
        font-size: 16px;
        color: #666666;
    }

    /* Chat area */
    .chat-container {
        max-width: 850px;
        margin: auto;
    }

    /* User message */
    .user-message {
        background: #111111;
        color: white;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0 10px auto;
        max-width: 75%;
        width: fit-content;
    }

    /* Assistant message */
    .assistant-message {
        background: white;
        color: #222222;
        padding: 16px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px auto 10px 0;
        max-width: 80%;
        width: fit-content;
        border: 1px solid #e5e5e5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Input */
    .stChatInput {
        padding-bottom: 20px;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Header
# -----------------------------

st.markdown(
    """
    <div class="main-header">
        <h1>🏍️ Jawa AI</h1>
        <p>Your AI assistant for Jawa motorcycles</p>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Load models
# -----------------------------

@st.cache_resource
def load_rag():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    return retriever, llm


retriever, llm = load_rag()


# -----------------------------
# Chat history
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.markdown("## 🏍️ Jawa AI")

    st.markdown(
        """
        Ask questions about:

        - Jawa motorcycles
        - Specifications
        - Engine details
        - Features
        - Models
        - Performance
        """
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption("Powered by")
    st.caption("FAISS + Hugging Face + Groq")


# -----------------------------
# Display chat history
# -----------------------------

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="user-message">
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="assistant-message">
                🏍️ <b>Jawa AI</b><br><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------
# Chat input
# -----------------------------

question = st.chat_input(
    "Ask something about Jawa motorcycles..."
)


if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display user message immediately
    st.markdown(
        f"""
        <div class="user-message">
            {question}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Retrieve documents
    with st.spinner("🔎 Searching the Jawa knowledge base..."):

        documents = retriever.invoke(question)

    # Create context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Prompt
    prompt = f"""
You are a helpful Jawa motorcycle assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not available in the context, say:

"I don't know based on the provided Jawa document."

Do not make up specifications or facts.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate response
    with st.spinner("🏍️ Generating answer..."):

        response = llm.invoke(prompt)

    answer = response.content

    # Save assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Display assistant response
    st.markdown(
        f"""
        <div class="assistant-message">
            🏍️ <b>Jawa AI</b><br><br>
            {answer}
        </div>
        """,
        unsafe_allow_html=True
    )