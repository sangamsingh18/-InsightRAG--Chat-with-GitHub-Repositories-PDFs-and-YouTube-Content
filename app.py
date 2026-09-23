import os
# Force transformers to only use PyTorch and skip TensorFlow/Keras import checks
os.environ["USE_TF"] = "0"
# Fix protobuf descriptor compilation mismatch
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Import our backend services
from src.github_loader import clone_repository, parse_github_url
from src.repo_parser import parse_repository
from src.chunking import chunk_files
from src.vectordb import save_to_vector_db, get_retriever, delete_repository_index
from src.utils import count_tokens, build_ascii_tree, clean_relative_path, log_indexed_source, safe_rmtree
from src.rag_chain import (
    ask_question_stream, 
    generate_summary, 
    generate_interview_questions, 
    generate_documentation, 
    generate_mermaid_diagram,
    explain_code_snippet,
    robust_retrieve,
    run_llm_with_fallback
)
from src.youtube_loader import load_youtube_transcript, extract_video_id
from src.document_loader import load_uploaded_file

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="InsightRAG - Chat with Repositories, PDFs, and YouTube Content",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Premium Dark CSS Styling
st.markdown("""
<style>
    /* Styling for Streamlit layout */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
        font-family: 'Inter', sans-serif;
    }
    
    /* Input field design */
    div[data-baseweb="input"] {
        background-color: #1f2833 !important;
        border-radius: 8px !important;
        border: 1px solid #2b3540 !important;
    }
    div[data-baseweb="input"] input {
        color: #ffffff !important;
    }
    
    /* Selectbox styling */
    div[data-baseweb="select"] {
        background-color: #1f2833 !important;
        border-radius: 8px !important;
    }
    
    /* Expander card design */
    .streamlit-expanderHeader {
        background-color: #1f2833 !important;
        border: 1px solid #2b3540 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    .streamlit-expanderContent {
        background-color: #0d1117 !important;
        border: 1px solid #2b3540 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 15px !important;
    }
    
    /* Tab headers custom colors */
    button[data-baseweb="tab"] {
        color: #8f94a5 !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
        transition: all 0.3s ease !important;
        border-radius: 8px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #66fcf1 !important;
        background-color: rgba(102, 252, 241, 0.1) !important;
        border-bottom: 2px solid #66fcf1 !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #66fcf1 !important;
        background-color: rgba(102, 252, 241, 0.05) !important;
    }
    
    /* Main gradient action button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #66fcf1 0%, #7c3aed 100%) !important;
        color: #0b0c10 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(102, 252, 241, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 252, 241, 0.5) !important;
    }
    
    /* Secondary or warning button action */
    div[data-testid="stSidebar"] div.stButton > button {
        background: #1f2833 !important;
        border: 1px solid #2b3540 !important;
        color: #c5c6c7 !important;
        box-shadow: none !important;
    }
    div[data-testid="stSidebar"] div.stButton > button:hover {
        background: #2b3540 !important;
        border-color: #66fcf1 !important;
        color: #ffffff !important;
    }
    
    /* Custom info banners / metrics dashboard */
    .metric-card {
        background: linear-gradient(135deg, #1f2833 0%, #121824 100%);
        border: 1px solid #2b3540;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #66fcf1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State variables
if "processed_sources" not in st.session_state:
    st.session_state.processed_sources = {}
if "selected_source_name" not in st.session_state:
    st.session_state.selected_source_name = None
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "query_to_run" not in st.session_state:
    st.session_state.query_to_run = None

# History-tracking lists for other tabs (to preserve previous generations)
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = {}
if "documentations" not in st.session_state:
    st.session_state.documentations = {}
if "diagrams" not in st.session_state:
    st.session_state.diagrams = {}

# Set the default LLM model
selected_model = "openai/gpt-oss-20b"
# Load API key strictly from environment configuration (.env)
groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
st.session_state.groq_api_key = groq_api_key

# Sidebar implementation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 25px; padding: 15px; background: rgba(31, 40, 51, 0.4); border-radius: 12px; border: 1px solid rgba(102, 252, 241, 0.15);">
        <h2 style="margin: 0; color: #66fcf1; font-family: 'Outfit', sans-serif; font-size: 1.7rem; font-weight: 800; text-shadow: 0 0 10px rgba(102, 252, 241, 0.35);">🧬 InsightRAG</h2>
        <p style="font-size: 0.78rem; color: #8f94a5; margin: 4px 0 0 0;">Multi-Source Context Analyst</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if Groq API key is present in environment
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        st.warning("⚠️ **GROQ_API_KEY** is missing or set to placeholder in your `.env` file. Please update the `.env` file to execute queries.")
    
    # 2. Source Loader Configuration
    st.markdown("### 📁 Choose RAG Source Mode")
    rag_mode = st.selectbox(
        "Source Mode",
        options=["📁 GitHub Repository", "🎥 YouTube Transcript", "📄 Uploaded Documents"]
    )
    
    if rag_mode == "📁 GitHub Repository":
        repo_url = st.text_input("GitHub Public URL", placeholder="https://github.com/owner/repo")
        col_proc, col_ref = st.columns(2)
        process_clicked = col_proc.button("Process Repo")
        refresh_clicked = col_ref.button("Refresh Repo")
        
        if process_clicked or refresh_clicked:
            if not groq_api_key:
                st.error("Please supply a Groq API Key.")
            elif not repo_url:
                st.error("Please provide a valid public GitHub URL.")
            else:
                owner, repo, clean_url = parse_github_url(repo_url)
                if not owner:
                    st.error("Invalid GitHub URL format.")
                else:
                    repo_display_name = f"repo_{owner}/{repo}"
                    with st.spinner("Processing repository..."):
                        try:
                            progress_bar = st.progress(0, text="Cloning repository...")
                            repo_path = clone_repository(clean_url, refresh=refresh_clicked)
                            
                            progress_bar.progress(25, text="Parsing source files...")
                            parsed_files, stats, tree = parse_repository(repo_path)
                            
                            total_tokens = sum(count_tokens(f["content"]) for f in parsed_files)
                            stats["total_tokens"] = total_tokens
                            
                            progress_bar.progress(50, text="Splitting code chunks...")
                            chunks = chunk_files(parsed_files)
                            stats["total_chunks"] = len(chunks)
                            
                            progress_bar.progress(75, text="Generating embeddings & indexing (ChromaDB)...")
                            save_to_vector_db(chunks, repo_display_name)
                            
                            log_indexed_source("GitHub Repository", clean_url)
                            safe_rmtree(repo_path)
                            if os.path.exists("repositories") and not os.listdir("repositories"):
                                safe_rmtree("repositories")
                            
                            progress_bar.progress(100, text="Indexing complete!")
                            st.session_state.processed_sources[repo_display_name] = {
                                "type": "GitHub Repository",
                                "name": f"{owner}/{repo}",
                                "stats": stats,
                                "tree": tree,
                                "parsed_files": parsed_files
                            }
                            st.session_state.selected_source_name = repo_display_name
                            time.sleep(1)
                            progress_bar.empty()
                            st.success(f"Indexed GitHub Repository: {owner}/{repo}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ingestion failed: {e}")
                            logger.error(f"GitHub Pipeline error: {e}", exc_info=True)

    elif rag_mode == "🎥 YouTube Transcript":
        yt_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        process_clicked = st.button("Process Video")
        
        if process_clicked:
            if not groq_api_key:
                st.error("Please supply a Groq API Key.")
            elif not yt_url:
                st.error("Please enter a YouTube video URL.")
            else:
                video_id = extract_video_id(yt_url)
                if not video_id:
                    st.error("Invalid YouTube URL.")
                else:
                    yt_display_name = f"yt_{video_id}"
                    with st.spinner("Fetching transcript and indexing..."):
                        try:
                            progress_bar = st.progress(10, text="Downloading YouTube captions...")
                            docs = load_youtube_transcript(yt_url)
                            
                            progress_bar.progress(40, text="Splitting transcript chunks...")
                            from langchain_text_splitters import RecursiveCharacterTextSplitter
                            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                            chunks = splitter.split_documents(docs)
                            
                            stats = {
                                "total_files": 1,
                                "total_folders": 0,
                                "total_chunks": len(chunks),
                                "total_lines": len(docs[0].page_content.splitlines()),
                                "total_tokens": count_tokens(docs[0].page_content)
                            }
                            
                            progress_bar.progress(70, text="Generating embeddings & indexing (ChromaDB)...")
                            save_to_vector_db(chunks, yt_display_name)
                            log_indexed_source("YouTube Video", yt_url)
                            
                            progress_bar.progress(100, text="Completed YouTube indexing!")
                            st.session_state.processed_sources[yt_display_name] = {
                                "type": "YouTube Video Transcript",
                                "name": f"YouTube Video: {video_id}",
                                "stats": stats,
                                "video_url": yt_url
                            }
                            st.session_state.selected_source_name = yt_display_name
                            time.sleep(1)
                            progress_bar.empty()
                            st.success(f"Indexed video transcript for ID: {video_id}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"YouTube processing failed: {e}")
                            logger.error(f"YouTube Pipeline error: {e}", exc_info=True)
                            
    elif rag_mode == "📄 Uploaded Documents":
        uploaded_files = st.file_uploader(
            "Select PDFs or Text files",
            type=["pdf", "txt", "md", "csv", "json", "yaml", "yml", "sql"],
            accept_multiple_files=True
        )
        doc_set_name = st.text_input("Name this Document Set", placeholder="my_study_material")
        process_clicked = st.button("Process Documents")
        
        if process_clicked:
            if not groq_api_key:
                st.error("Please supply a Groq API Key.")
            elif not uploaded_files:
                st.error("Please upload at least one document.")
            elif not doc_set_name:
                st.error("Please specify a name for the document set.")
            else:
                import re
                clean_set_name = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_set_name.strip().lower())
                doc_display_name = f"doc_{clean_set_name}"
                
                with st.spinner("Reading and indexing files..."):
                    try:
                        progress_bar = st.progress(10, text="Reading files...")
                        all_docs = []
                        for f in uploaded_files:
                            all_docs.extend(load_uploaded_file(f))
                            
                        progress_bar.progress(40, text="Splitting text contents...")
                        from langchain_text_splitters import RecursiveCharacterTextSplitter
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
                        chunks = splitter.split_documents(all_docs)
                        
                        stats = {
                            "total_files": len(uploaded_files),
                            "total_folders": 0,
                            "total_chunks": len(chunks),
                            "total_lines": sum(len(d.page_content.splitlines()) for d in all_docs),
                            "total_tokens": sum(count_tokens(d.page_content) for d in all_docs)
                        }
                        
                        progress_bar.progress(70, text="Generating embeddings & indexing (ChromaDB)...")
                        save_to_vector_db(chunks, doc_display_name)
                        log_indexed_source("Uploaded Documents", f"Name: {doc_set_name} (Files: {', '.join([f.name for f in uploaded_files])})")
                        
                        progress_bar.progress(100, text="Completed document indexing!")
                        st.session_state.processed_sources[doc_display_name] = {
                            "type": "Uploaded Documents",
                            "name": f"Docs: {doc_set_name}",
                            "stats": stats,
                            "files_list": [f.name for f in uploaded_files]
                        }
                        st.session_state.selected_source_name = doc_display_name
                        time.sleep(1)
                        progress_bar.empty()
                        st.success(f"Indexed {len(uploaded_files)} files as: {doc_set_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Document processing failed: {e}")
                        logger.error(f"Document Pipeline error: {e}", exc_info=True)
                        
    st.markdown("---")
    
    # 3. Context Switcher for active sources
    if st.session_state.processed_sources:
        st.markdown("### 🔀 Active Knowledge Base")
        source_options = list(st.session_state.processed_sources.keys())
        
        def get_label(key):
            item = st.session_state.processed_sources[key]
            return f"[{item['type'][:4]}] {item['name']}"
            
        default_index = source_options.index(st.session_state.selected_source_name) if st.session_state.selected_source_name in source_options else 0
        selected_source = st.selectbox(
            "Select Active Base",
            options=source_options,
            format_func=get_label,
            index=default_index
        )
        if selected_source != st.session_state.selected_source_name:
            st.session_state.selected_source_name = selected_source
            st.rerun()
            
        if st.button("Delete Selected Index"):
            del_src = st.session_state.selected_source_name
            with st.spinner("Deleting database..."):
                delete_repository_index(del_src)
                st.session_state.processed_sources.pop(del_src, None)
                st.session_state.chat_histories.pop(del_src, None)
                st.session_state.summaries.pop(del_src, None)
                st.session_state.interview_questions.pop(del_src, None)
                st.session_state.documentations.pop(del_src, None)
                st.session_state.diagrams.pop(del_src, None)
                
                if st.session_state.processed_sources:
                    st.session_state.selected_source_name = list(st.session_state.processed_sources.keys())[0]
                else:
                    st.session_state.selected_source_name = None
                st.success("Deleted active index.")
                st.rerun()
    else:
        st.info("No knowledge sources indexed yet. Use the selector above to ingest content.")

# Main Panel layout
st.markdown("""
<div style="background: linear-gradient(90deg, #1f2833 0%, #111a24 50%, #0d121c 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px; border: 1px solid rgba(102, 252, 241, 0.2); box-shadow: 0 10px 30px rgba(0,0,0,0.4);">
    <h1 style="color: #66fcf1; margin: 0; font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 2.2rem; letter-spacing: -0.5px;">InsightRAG</h1>
    <p style="color: rgba(255, 255, 255, 0.75); margin: 6px 0 0 0; font-size: 1.05rem; font-weight: 300;">Chat with GitHub Repositories, PDFs, and YouTube Content</p>
</div>
""", unsafe_allow_html=True)

# Main rendering block
if st.session_state.selected_source_name:
    active_source = st.session_state.selected_source_name
    source_data = st.session_state.processed_sources[active_source]
    stats = source_data["stats"]
    source_type = source_data["type"]
    source_name = source_data["name"]
    
    # Display statistics dashboard
    st.markdown(f"### 📊 Context Stats: `{source_name}` ({source_type})")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#8f94a5;">Folders</span><br/>
            <strong style="font-size:1.8rem; color:#66fcf1;">{stats['total_folders']}</strong>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#8f94a5;">Loaded Files</span><br/>
            <strong style="font-size:1.8rem; color:#66fcf1;">{stats['total_files']}</strong>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#8f94a5;">Total Chunks</span><br/>
            <strong style="font-size:1.8rem; color:#66fcf1;">{stats['total_chunks']}</strong>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#8f94a5;">Total Lines</span><br/>
            <strong style="font-size:1.8rem; color:#66fcf1;">{stats['total_lines']:,}</strong>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#8f94a5;">Est. Tokens</span><br/>
            <strong style="font-size:1.8rem; color:#66fcf1;">{stats['total_tokens']:,}</strong>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Configure dynamic tabs based on active source type
    if source_type == "GitHub Repository":
        tab_titles = [
            "💬 Conversational Q&A",
            "📋 Project Summaries",
            "📂 Explorer & Code Explainer",
            "🎯 Technical Interview Prep",
            "📝 Document Generator",
            "🧬 Architectural Diagrams"
        ]
        tabs = st.tabs(tab_titles)
        tab_chat, tab_summary, tab_explorer, tab_interview, tab_docs, tab_arch = tabs
    else:
        # For non-code sources (YouTube, uploaded documents), only show Q&A and summary
        tab_titles = [
            "💬 Conversational Q&A",
            "📋 Project Summaries"
        ]
        tabs = st.tabs(tab_titles)
        tab_chat, tab_summary = tabs
        tab_explorer = tab_interview = tab_docs = tab_arch = None

    # ------------------ TAB 1: Conversational Chat ------------------
    with tab_chat:
        st.markdown(f"### 💬 Conversational Dialogue with: `{source_name}`")
        st.caption(f"Active base: **{source_type}** | Brain Model: **{selected_model}**")
        
        # Initialize chat histories
        if active_source not in st.session_state.chat_histories:
            st.session_state.chat_histories[active_source] = []
            
        history = st.session_state.chat_histories[active_source]
        
        # Render history
        for role, message_text in history:
            with st.chat_message(role):
                st.markdown(message_text)
                
        # Run active query if present in session state
        active_query = st.session_state.query_to_run
        if active_query:
            st.session_state.query_to_run = None  # Clear trigger
            
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                sources_placeholder = st.empty()
                
                full_response = ""
                sources_list = []
                
                # Streaming loop
                for chunk in ask_question_stream(
                    query=active_query,
                    history=history[:-1] if (history and history[-1][0] == "user") else history,
                    source_name=active_source,
                    source_type=source_type,
                    api_key=groq_api_key,
                    model_name=selected_model
                ):
                    if chunk["type"] == "sources":
                        sources_list = chunk["data"]
                    elif chunk["type"] == "content":
                        full_response += chunk["data"]
                        response_placeholder.markdown(full_response + "▌")
                    elif chunk["type"] == "error":
                        st.error(chunk["data"])
                        
                response_placeholder.markdown(full_response)
                
                # Render sources
                if sources_list:
                    sources_md = "#### 📚 Sources Cited:\n"
                    for s in sources_list:
                        if source_type == "GitHub Repository":
                            sources_md += f"- **{s['file_name']}** (Path: `{s['file_path']}`, Lang: *{s['language']}*)\n"
                        elif source_type == "YouTube Video Transcript":
                            sources_md += f"- Transcript segment around timestamp references.\n"
                            break
                        else:  # Uploaded Docs
                            sources_md += f"- **{s['file_name']}**\n"
                    sources_placeholder.markdown("---")
                    sources_placeholder.markdown(sources_md)
                    
                history.append(("assistant", full_response))
                st.rerun()
                
        # Render action buttons below the last assistant message
        if history and history[-1][0] == "assistant":
            col_reg, col_clr, _ = st.columns([1.8, 1.8, 5])
            if col_reg.button("🔄 Regenerate Answer", key="regen_last_resp_btn"):
                user_msg_idx = -1
                for idx in range(len(history) - 1, -1, -1):
                    if history[idx][0] == "user":
                        user_msg_idx = idx
                        break
                
                if user_msg_idx != -1:
                    last_query = history[user_msg_idx][1]
                    st.session_state.chat_histories[active_source] = history[:user_msg_idx + 1]
                    st.session_state.query_to_run = last_query
                    st.rerun()
                    
            if col_clr.button("🧹 Clear Chat History", key="clear_chat_button"):
                st.session_state.chat_histories[active_source] = []
                st.rerun()

        # Handle bottom input
        user_input = st.chat_input("Ask a question about this content...")
        if user_input:
            if not groq_api_key:
                st.error("Please configure your Groq API Key in the sidebar.")
            else:
                history.append(("user", user_input))
                st.session_state.query_to_run = user_input
                st.rerun()

    # ------------------ TAB 2: Project Summaries ------------------
    with tab_summary:
        st.markdown("### 📋 Automated Project Analysis & Summaries")
        
        # Initialize list cache for summaries
        if active_source not in st.session_state.summaries:
            st.session_state.summaries[active_source] = []
            
        summaries_list = st.session_state.summaries[active_source]
        
        # Render list of generated summaries sequentially
        for idx, text in enumerate(summaries_list):
            st.markdown(f"#### 📅 Summary Version #{idx+1}")
            st.markdown(text)
            st.markdown("---")
            
        # Controls for generating another version
        col1, _ = st.columns([2, 5])
        button_label = "⚡ Generate Summary" if not summaries_list else "🔄 Generate Another Version"
        if col1.button(button_label, key="gen_summary_btn"):
            if not groq_api_key:
                st.error("Groq API Key is required.")
            else:
                with st.spinner("Generating summary..."):
                    # For non-code sources, summarize using direct query RAG
                    if source_type != "GitHub Repository":
                        search_query = "overall summary core concept key takeaways main ideas details"
                        docs = robust_retrieve(active_source, search_query, top_k=10)
                        context = "\n".join([f"--- Section {i+1} ---\n{d.page_content}" for i, d in enumerate(docs)])
                        
                        prompt = f"""You are an Expert Summarizer. Generate a comprehensive summary of the provided text.
                        Provide:
                        1. A high-level Executive Summary (main message, purpose).
                        2. Key Takeaways and Major Themes.
                        3. Detailed Breakdown of content.
                        
                        Source Content:
                        {context}
                        """
                        
                        # Wrap LLM call in fallback logic
                        result = run_llm_with_fallback(
                            lambda llm: llm.invoke(prompt).content, 
                            groq_api_key, 
                            selected_model
                        )
                    else:
                        result = generate_summary(active_source, groq_api_key, selected_model)
                        
                    summaries_list.append(result)
                    st.rerun()

    # ------------------ TAB 3: File Explorer (Code Repo Only) ------------------
    if tab_explorer:
        with tab_explorer:
            st.markdown("### 📂 Repository File Explorer")
            parsed_files = source_data["parsed_files"]
            tree = source_data["tree"]
            
            with st.expander("📁 Visual Directory Tree Viewer (ASCII)", expanded=False):
                ascii_tree = build_ascii_tree(tree)
                st.code(ascii_tree, language="text")
                
            st.markdown("---")
            st.markdown("### 🔍 Code Explainer")
            st.caption("Select a source file to view its code and generate a line-by-line mechanical explanation.")
            
            file_options = {f["file_path"]: f for f in parsed_files}
            selected_file_path = st.selectbox(
                "Select File to Inspect",
                options=list(file_options.keys())
            )
            
            if selected_file_path:
                file_meta = file_options[selected_file_path]
                st.markdown(f"**Language**: `{file_meta['language']}` | **Size**: `{file_meta['size_bytes']:,} bytes` | **Lines**: `{file_meta['lines_count']}`")
                
                lang_syntax_map = {
                    'Python': 'python', 'JavaScript': 'javascript', 'JavaScript React': 'javascript',
                    'TypeScript': 'typescript', 'TypeScript React': 'typescript', 'HTML': 'html',
                    'CSS': 'css', 'SCSS': 'scss', 'JSON': 'json', 'YAML': 'yaml',
                    'Markdown': 'markdown', 'SQL': 'sql', 'C++': 'cpp', 'C': 'c',
                    'C#': 'csharp', 'Go': 'go', 'Rust': 'rust', 'Java': 'java',
                    'PHP': 'php', 'Dockerfile': 'dockerfile'
                }
                syntax = lang_syntax_map.get(file_meta['language'], 'text')
                
                with st.expander("📄 View Source Code", expanded=True):
                    st.code(file_meta["content"], language=syntax)
                    
                if st.button("💡 Explain Code Line-by-Line", key="explain_code_btn"):
                    if not groq_api_key:
                        st.error("Groq API Key is required.")
                    else:
                        with st.spinner("Analyzing code structure..."):
                            code_to_explain = file_meta["content"]
                            if len(code_to_explain) > 10000:
                                code_to_explain = code_to_explain[:10000] + "\n\n... [Content Truncated] ..."
                            explanation = explain_code_snippet(
                                file_path=file_meta["file_path"],
                                language=file_meta["language"],
                                code=code_to_explain,
                                api_key=groq_api_key,
                                model_name=selected_model
                            )
                            st.markdown("### 🔬 Line-by-Line Breakdown")
                            st.markdown(explanation)

    # ------------------ TAB 4: Technical Interview Prep (Code Repo Only) ------------------
    if tab_interview:
        with tab_interview:
            st.markdown("### 🎯 Technical Interview Question Generator")
            st.caption("Generates engineering questions tailored to architectural files.")
            
            diff = st.selectbox("Select Target Difficulty", options=["Beginner", "Intermediate", "Advanced"])
            q_count = st.slider("Number of Questions", min_value=5, max_value=15, value=5)
            
            cache_key = f"{active_source}_{diff}_{q_count}"
            if cache_key not in st.session_state.interview_questions:
                st.session_state.interview_questions[cache_key] = []
                
            questions_list = st.session_state.interview_questions[cache_key]
            
            # Render list of generated questions sequentially
            for idx, text in enumerate(questions_list):
                with st.expander(f"🎯 Interview Question Set #{idx+1}", expanded=(idx == len(questions_list) - 1)):
                    st.markdown(text)
                    
            # Buttons
            button_label = "🎯 Generate Questions" if not questions_list else "🔄 Generate Another Set"
            if st.button(button_label, key="gen_q_btn"):
                if not groq_api_key:
                    st.error("Groq API Key is required.")
                else:
                    with st.spinner("Generating interview questions..."):
                        questions = generate_interview_questions(
                            repository_name=active_source,
                            difficulty=diff,
                            count=q_count,
                            api_key=groq_api_key,
                            model_name=selected_model
                        )
                        questions_list.append(questions)
                        st.rerun()

    # ------------------ TAB 5: Documentation Generator (Code Repo Only) ------------------
    if tab_docs:
        with tab_docs:
            st.markdown("### 📝 Auto-Generate Technical Documentation")
            st.caption("Compiles technical reference documents.")
            
            doc_category = st.radio(
                "Select Documentation Type",
                options=["README", "API Documentation", "Architecture Documentation"],
                horizontal=True
            )
            
            cache_key = f"{active_source}_{doc_category}"
            if cache_key not in st.session_state.documentations:
                st.session_state.documentations[cache_key] = []
                
            docs_list = st.session_state.documentations[cache_key]
            
            # Render all generated document versions
            for idx, text in enumerate(docs_list):
                with st.expander(f"📝 {doc_category} Version #{idx+1}", expanded=(idx == len(docs_list) - 1)):
                    st.markdown(text)
                    st.download_button(
                        label=f"💾 Download Version #{idx+1}",
                        data=text,
                        file_name=f"{doc_category.lower().replace(' ', '_')}_v{idx+1}.md",
                        mime="text/markdown"
                    )
                    
            # Buttons
            button_label = "📝 Compile Documentation" if not docs_list else "🔄 Compile Another Version"
            if st.button(button_label, key="gen_doc_btn"):
                if not groq_api_key:
                    st.error("Groq API Key is required.")
                else:
                    with st.spinner(f"Compiling {doc_category}..."):
                        docs_output = generate_documentation(
                            repository_name=active_source,
                            doc_type=doc_category,
                            api_key=groq_api_key,
                            model_name=selected_model
                        )
                        docs_list.append(docs_output)
                        st.rerun()

    # ------------------ TAB 6: Architectural Diagrams (Code Repo Only) ------------------
    if tab_arch:
        with tab_arch:
            st.markdown("### 🧬 Mermaid Architecture Diagram Generator")
            st.caption("Generates class, flow, and sequence diagrams.")
            
            diagram_category = st.selectbox(
                "Select Diagram Category",
                options=["Class Diagram", "Sequence Diagram", "Flow Diagram"]
            )
            
            cache_key = f"{active_source}_{diagram_category}"
            if cache_key not in st.session_state.diagrams:
                st.session_state.diagrams[cache_key] = []
                
            diagrams_list = st.session_state.diagrams[cache_key]
            
            # Render all generated diagrams
            for idx, code in enumerate(diagrams_list):
                with st.expander(f"📊 Diagram Version #{idx+1}", expanded=(idx == len(diagrams_list) - 1)):
                    st.markdown("#### Rendered Diagram")
                    st.markdown(code)
                    st.markdown("#### Mermaid Code Source")
                    st.code(code, language="mermaid")
                    
            # Buttons
            button_label = "🧬 Draw Diagram" if not diagrams_list else "🔄 Draw Another Diagram"
            if st.button(button_label, key="gen_diagram_btn"):
                if not groq_api_key:
                    st.error("Groq API Key is required.")
                else:
                    with st.spinner("Plotting Mermaid diagram..."):
                        diagram = generate_mermaid_diagram(
                            repository_name=active_source,
                            diagram_type=diagram_category,
                            api_key=groq_api_key,
                            model_name=selected_model
                        )
                        diagrams_list.append(diagram)
                        st.rerun()
else:
    # Warm welcome state when no source is loaded
    st.markdown("""
    <div style="background-color: #1f2833; border: 1px dashed #66fcf1; border-radius: 12px; padding: 40px; text-align: center; margin-top: 40px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
        <span style="font-size: 3rem;">👋</span>
        <h3 style="color: #66fcf1; margin-top: 15px;">Welcome to InsightRAG Portal</h3>
        <p style="color: #c5c6c7; max-width: 650px; margin: 10px auto 25px auto; font-size: 0.95rem; line-height: 1.6;">
            InsightRAG is an all-in-one retrieval agent. 
            Choose your ingestion mode in the sidebar to process a <strong>GitHub Repository</strong>, extract transcription from a <strong>YouTube Video</strong>, or index uploaded <strong>PDF & Text Documents</strong>.
        </p>
        <div style="display: inline-flex; gap: 20px; color: #8f94a5; font-size: 0.85rem;">
            <span>✔️ GitHub Repositories</span>
            <span>✔️ YouTube Transcripts</span>
            <span>✔️ PDF / TXT Documents</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
