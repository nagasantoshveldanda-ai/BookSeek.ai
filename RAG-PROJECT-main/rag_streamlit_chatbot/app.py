import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import traceback
from utils.data_ingest import load_and_split
from utils.vector_store import build_vector_db, save_vector_db, load_vector_db, add_to_vector_db, get_source_documents
from utils.rag_chain import create_rag_chain, OpenRouterLLM
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="BookSeek.ai Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
    }

    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px #aaa;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in-out;
        position: relative;
    }

    .user-message {
        background-color: #DCF8C6;
        margin-left: auto;
        width: 80%;
        border-radius: 1rem 1rem 0 1rem;
    }

    .bot-message {
        background-color: #ECE5DD;
        width: 80%;
        border-radius: 1rem 1rem 1rem 0;
    }
    
    .chat-message strong {
        font-weight: 700;
    }

    .sidebar .st-emotion-cache-10oheav {
        padding-top: 2rem;
    }
    
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
body, .stApp {
    background-image: url('https://www.imghippo.com/i/wmd4483PA.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state["conversations"] = {} # { "conversation_name": [ (q,a), ... ] }
if "current_conversation" not in st.session_state:
    st.session_state["current_conversation"] = "new"
    st.session_state["conversations"]["new"] = []

if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None
if "document_processed" not in st.session_state:
    st.session_state["document_processed"] = False
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# Directory to store vector DBs
VECTOR_DB_DIR = "vector_dbs"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Auto-load existing vector DB on startup
COMBINED_DB_PATH = os.path.join(VECTOR_DB_DIR, "combined_vector_db")
if os.path.exists(os.path.join(COMBINED_DB_PATH, "chroma-collections.parquet")):
    try:
        st.session_state["vectorstore"] = load_vector_db(COMBINED_DB_PATH)
        st.session_state["rag_chain"] = create_rag_chain(st.session_state["vectorstore"])
        st.session_state["document_processed"] = True
        st.sidebar.success("📚 Loaded existing vector database.")
        
        # Display loaded documents
        loaded_docs = get_source_documents(st.session_state["vectorstore"])
        if loaded_docs:
            st.sidebar.markdown("---")
            st.sidebar.markdown('<h3 style="color: #4A4A4A; font-size: 1.2rem; margin-bottom: 1rem;">Loaded Documents</h3>', unsafe_allow_html=True)
            for doc_name in loaded_docs:
                st.sidebar.markdown(f"- {os.path.basename(doc_name)}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading existing vector database: {str(e)}")
        st.session_state["document_processed"] = False

# Sidebar
st.sidebar.markdown('<h2 style="color: #4A4A4A; font-size: 1.8rem; margin-bottom: 0.5rem;">BookSeek.AI</h2>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="color: #888888; font-size: 0.9rem; margin-bottom: 2rem;">learn From AI</p>', unsafe_allow_html=True)


# File uploader (hidden visually, triggered by custom div)
uploaded_files = st.sidebar.file_uploader(
    " ", # Empty label to hide default text
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader",
    help="Add PDFs, notes, or study materials"
)





# API Key check
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    st.sidebar.error("⚠️ OPENROUTER_API_KEY not found in environment variables!")
    st.sidebar.info("Please add your OPENROUTER API key to the .env file")

# Document processing
if uploaded_files:
    # Use a fixed directory for combined vector DB for simplicity
    persist_directory = os.path.join(VECTOR_DB_DIR, "combined_vector_db")
    
    # Check if the current set of uploaded files has already been processed
    # This is a simple check; a more robust solution would involve hashing file contents
    uploaded_file_names = sorted([f.name for f in uploaded_files])
    if "last_processed_files" not in st.session_state or st.session_state["last_processed_files"] != uploaded_file_names:
        st.session_state["document_processed"] = False # Force re-processing if files changed
        st.session_state["last_processed_files"] = uploaded_file_names

    if not st.session_state["document_processed"]:
        try:
            with st.sidebar.container():
                st.info("📄 Processing documents...")
                progress_bar = st.progress(0)
                
                all_docs = []
                temp_file_paths = []

                for i, uploaded_file in enumerate(uploaded_files):
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                        temp_file_paths.append(tmp_path)

                    progress_bar.progress(int((i + 1) / len(uploaded_files) * 25))
                    
                    docs = load_and_split(tmp_path)
                    all_docs.extend(docs)
                
                progress_bar.progress(50)

                # Check if ChromaDB already exists for this combined set
                chroma_exists = os.path.exists(os.path.join(persist_directory, "chroma-collections.parquet"))
                if chroma_exists:
                    vectorstore = load_vector_db(persist_directory)
                    st.sidebar.info("♻️ Loaded existing vector database.")
                    progress_bar.progress(75)
                else:
                    vectorstore = build_vector_db(all_docs, persist_directory)
                    save_vector_db(vectorstore) # Save the vector DB
                    st.sidebar.info("💾 Vector database built and saved.")
                    progress_bar.progress(75)
                
                st.session_state["vectorstore"] = vectorstore
                rag_chain = create_rag_chain(vectorstore)
                st.session_state["rag_chain"] = rag_chain
                progress_bar.progress(100)

                # Clean up temporary files
                for tmp_path in temp_file_paths:
                    os.unlink(tmp_path)

                st.session_state["document_processed"] = True
                st.sidebar.success(f"✅ {len(uploaded_files)} document(s) processed and added to the database!")
                st.rerun()

        except Exception as e:
            st.sidebar.error(f"❌ Error processing documents: {str(e)}")
            st.sidebar.error("Please check your files and try again.")
            # Clean up temporary files in case of error
            for tmp_path in temp_file_paths:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# Main content area
st.markdown('<h2 style="color: #4A4A4A; font-size: 1.8rem; margin-bottom: 0.5rem;">Learning Assistant</h2>', unsafe_allow_html=True)
st.markdown('<p style="color: #888888; font-size: 0.9rem; margin-bottom: 2rem;">Ask me anything about your studies</p>', unsafe_allow_html=True)

if not st.session_state["document_processed"]:
    st.markdown("""
    <div style="text-align: center; margin-top: 5rem;">
        <h1 style="color: #4A4A4A; margin-top: 2rem;"> <span style="color: #6c5ce7;">BookSeek.AI</span></h1>
        <p style="color: #888888; font-size: 1.1rem; max-width: 600px; margin: 1rem auto 0 auto;">
            learn with Ai to get personalized explanations tailored to your learning style.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Placeholder for chat input when no document is processed
    st.text_input(
        " ",
        placeholder="Ask me anything about your studies...",
        key="initial_user_input",
        disabled=True
    )
    st.markdown("""
    <style>
        .stTextInput [data-testid="stFormSubmitButton"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

else:
    # Chat interface
    # Display chat history
    chat_history = st.session_state["conversations"].get(st.session_state["current_conversation"], [])
    for i, (question, answer) in enumerate(chat_history):
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🙋 You:</strong> {question}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Assistant:</strong> {answer}
        </div>
        """, unsafe_allow_html=True)

    # Query input at the bottom
    user_question = st.chat_input(
        "Ask me anything about your studies...",
        key="chat_input"
    )

    # Process query
    if user_question and st.session_state.get("rag_chain"):
        # If this is the first message in a new conversation, use it as the name
        if st.session_state["current_conversation"] == "new":
            # Rename the conversation with the first question
            st.session_state["conversations"][user_question] = st.session_state["conversations"].pop("new")
            st.session_state["current_conversation"] = user_question

        try:
            with st.spinner("🤔 Thinking..."):
                result = st.session_state["rag_chain"]({"query": user_question})
                answer = result["result"]

                # Add to chat history
                st.session_state["conversations"][st.session_state["current_conversation"]].append((user_question, answer))

                # Add conversation to vector store
                qa_docs = [
                    Document(page_content=f"User Question: {user_question}"),
                    Document(page_content=f"Assistant Answer: {answer}")
                ]
                add_to_vector_db(st.session_state["vectorstore"], qa_docs)

                # Display latest response
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🙋 You:</strong> {user_question}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant:</strong> {answer}
                </div>
                """, unsafe_allow_html=True)

                # Show source documents if available
                if "source_documents" in result and result["source_documents"]:
                    with st.expander("📖 View Sources"):
                        for i, doc in enumerate(result["source_documents"][:3]):
                            st.markdown(f"**Source {i + 1}:**")
                            st.text(doc.page_content[:300] + "...")
        except Exception as e:
            st.error(f"❌ Error processing your question: {str(e)}")
            st.error("Please try rephrasing your question or check your API key.")


