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

with st.sidebar:
    theme_mode = st.radio(
        "Appearance",
        options=["Light", "Dark"],
        horizontal=True,
        key="theme_mode",
    )

theme_palette = {
    "Light": {
        "background": "#f5f5f5",
        "surface": "#ffffff",
        "text": "#222222",
        "heading": "#111111",
        "muted": "#666666",
        "border": "#e5e5e5",
        "user_background": "#111111",
        "user_text": "#ffffff",
        "input_background": "#ffffff",
    },
    "Dark": {
        "background": "#15171c",
        "surface": "#242832",
        "text": "#f1f3f5",
        "heading": "#ffffff",
        "muted": "#b0b7c3",
        "border": "#3a404c",
        "user_background": "#d6a84f",
        "user_text": "#171717",
        "input_background": "#242832",
    },
}
palette = theme_palette[theme_mode]


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: {"dark" if theme_mode == "Dark" else "light"};
        --page-bg: {palette["background"]};
        --surface: {palette["surface"]};
        --text: {palette["text"]};
        --heading: {palette["heading"]};
        --muted: {palette["muted"]};
        --border: {palette["border"]};
        --user-background: {palette["user_background"]};
        --user-text: {palette["user_text"]};
        --input-background: {palette["input_background"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>

    /* Page, content, and fixed bottom surfaces */
    html,
    body,
    #root,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: var(--page-bg) !important;
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: var(--surface) !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    [data-testid="stSidebar"] button {
        background: var(--page-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    [data-testid="stSidebar"] input[type="radio"] {
        accent-color: #d6a84f;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 25px 0 10px 0;
    }

    .main-header h1 {
        font-size: 38px;
        margin-bottom: 5px;
        color: var(--heading);
    }

    .main-header p {
        font-size: 16px;
        color: var(--muted);
    }

    /* Chat area */
    .chat-container {
        max-width: 850px;
        margin: auto;
    }

    /* User message */
    .user-message {
        background: var(--user-background);
        color: var(--user-text);
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0 10px auto;
        max-width: 75%;
        width: fit-content;
    }

    /* Assistant message */
    .assistant-message {
        background: var(--surface);
        color: var(--text);
        padding: 16px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px auto 10px 0;
        max-width: 80%;
        width: fit-content;
        border: 1px solid var(--border);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        padding: 8px 0 20px;
        background: var(--page-bg) !important;
    }

    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] div[data-baseweb="textarea"] {
        background: var(--input-background) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"]:focus-within > div {
        border-color: #d6a84f !important;
        box-shadow: 0 0 0 2px rgba(214, 168, 79, 0.18) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--text) !important;
        border: 0 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted);
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    .stAppDeployButton,
    [data-testid="stAppDeployButton"] {
        display: none !important;
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

if "recent_chats" not in st.session_state:
    st.session_state.recent_chats = []


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

    st.markdown("### 🕘 Recent chats")

    if st.session_state.recent_chats:
        for index, chat in enumerate(reversed(st.session_state.recent_chats)):
            chat_index = len(st.session_state.recent_chats) - index - 1
            title = chat["question"].strip().replace("\n", " ")
            if len(title) > 34:
                title = f"{title[:31]}..."

            with st.expander(f"{index + 1}. {title}"):
                st.markdown("**You asked**")
                st.write(chat["question"])
                st.markdown("**Jawa AI replied**")
                st.write(chat["answer"])
                if st.button(
                    "🗑️ Delete this chat",
                    key=f"delete_recent_chat_{chat_index}",
                    use_container_width=True,
                ):
                    del st.session_state.recent_chats[chat_index]
                    st.rerun()
    else:
        st.caption("Your last three question-and-answer chats will appear here.")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()

    st.divider()



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

    st.session_state.recent_chats.append(
        {
            "question": question,
            "answer": answer,
        }
    )
    st.session_state.recent_chats = st.session_state.recent_chats[-3:]

    st.rerun()