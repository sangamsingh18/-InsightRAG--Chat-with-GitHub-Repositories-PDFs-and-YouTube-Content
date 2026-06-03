import logging
from typing import Generator, List, Dict, Any, Tuple
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.vectordb import get_retriever, get_vector_db
from src.prompts import (
    CHAT_SYSTEM_PROMPT_REPO,
    CHAT_SYSTEM_PROMPT_YT,
    CHAT_SYSTEM_PROMPT_DOC,
    SUMMARY_PROMPT,
    INTERVIEW_PROMPT,
    DOCS_PROMPT,
    MERMAID_PROMPT,
    EXPLAINER_PROMPT
)

# Setup logging
logger = logging.getLogger(__name__)

# Fallback configuration
FALLBACK_MODEL = "llama-3.1-8b-instant"

def robust_retrieve(source_name: str, query: str, top_k: int = 8) -> List[Any]:
    """
    Robustly retrieves context documents for the query.
    1. Attempts MMR (Maximal Marginal Relevance) retrieval for diversity.
    2. If MMR returns fewer than 2 documents, falls back to direct similarity search.
    """
    docs = []
    try:
        retriever = get_retriever(source_name, top_k=top_k)
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents using MMR for: '{query}'")
    except Exception as e:
        logger.warning(f"MMR retrieval failed for {source_name}: {e}. Trying similarity fallback.")
        
    if len(docs) < 2:
        try:
            db = get_vector_db(source_name)
            docs = db.similarity_search(query, k=top_k)
            logger.info(f"Retrieved {len(docs)} documents using Similarity fallback for: '{query}'")
        except Exception as e:
            logger.error(f"Similarity Search fallback failed for {source_name}: {e}")
            
    return docs

def rephrase_query(
    query: str, 
    history: List[Tuple[str, str]], 
    api_key: str, 
    model_name: str
) -> str:
    """
    Uses the Groq LLM to rephrase a user query containing context references
    into a standalone search query based on chat history.
    
    Includes a fallback model check.
    """
    if not history:
        return query
        
    messages = [
        SystemMessage(content=(
            "You are a technical search assistant. Your task is to rephrase the user's latest query "
            "into a standalone search query that incorporates details from the chat history. "
            "Respond ONLY with the rephrased standalone query. Do not add introductory phrases, "
            "markdown styling, or extra chat text."
        ))
    ]
    
    for role, content in history[-6:]:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
            
    messages.append(HumanMessage(content=f"Rephrase this latest query as a standalone search query: {query}"))
    
    # Try primary model first, fallback on rate limits
    try:
        llm = ChatGroq(model=model_name, groq_api_key=api_key, temperature=0.1)
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "rate_limit" in err_msg.lower():
            logger.warning(f"Rephrase rate limited on {model_name}. Retrying with {FALLBACK_MODEL}.")
            try:
                llm = ChatGroq(model=FALLBACK_MODEL, groq_api_key=api_key, temperature=0.1)
                response = llm.invoke(messages)
                return response.content.strip()
            except Exception:
                pass
        logger.warning(f"Query rephrase failed: {e}. Utilizing original query.")
        return query

def ask_question_stream(
    query: str,
    history: List[Tuple[str, str]],
    source_name: str,
    source_type: str,
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile"
) -> Generator[Dict[str, Any], None, None]:
    """
    Runs RAG QA streaming. Automatically switches to fallback model if 429 rate limit occurs.
    """
    search_query = rephrase_query(query, history, api_key, model_name)
    docs = robust_retrieve(source_name, search_query, top_k=8)
        
    sources = []
    seen_sources = set()
    for doc in docs:
        path = doc.metadata.get("file_path", doc.metadata.get("source", "Unknown"))
        name = doc.metadata.get("file_name", "Unknown")
        lang = doc.metadata.get("language", "Unknown")
        if path not in seen_sources:
            seen_sources.add(path)
            sources.append({"file_name": name, "file_path": path, "language": lang})
            
    yield {"type": "sources", "data": sources}
    
    context = ""
    for idx, doc in enumerate(docs):
        context += f"\n--- Source Chunk {idx+1} ({doc.metadata.get('file_name', 'Source')}) ---\n{doc.page_content}\n"
        
    if source_type == "GitHub Repository":
        system_prompt_content = CHAT_SYSTEM_PROMPT_REPO.format(context=context)
    elif source_type == "YouTube Video Transcript":
        system_prompt_content = CHAT_SYSTEM_PROMPT_YT.format(context=context)
    else:
        system_prompt_content = CHAT_SYSTEM_PROMPT_DOC.format(context=context)
        
    messages = [SystemMessage(content=system_prompt_content)]
    
    for role, content in history[-10:]:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=query))
    
    # Try primary LLM stream
    try:
        llm = ChatGroq(model=model_name, groq_api_key=api_key, temperature=0.2, streaming=True)
        for chunk in llm.stream(messages):
            yield {"type": "content", "data": chunk.content}
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "rate_limit" in err_msg.lower():
            logger.warning(f"QA Stream rate limited on {model_name}. Retrying with {FALLBACK_MODEL}.")
            yield {
                "type": "content", 
                "data": "\n\n⚠️ *[Groq 70B Rate Limit exceeded. Switched dynamically to fallback model Llama 3.1 8B...]*\n\n"
            }
            try:
                llm = ChatGroq(model=FALLBACK_MODEL, groq_api_key=api_key, temperature=0.2, streaming=True)
                for chunk in llm.stream(messages):
                    yield {"type": "content", "data": chunk.content}
            except Exception as e_fallback:
                yield {"type": "error", "data": f"Groq fallback error: {str(e_fallback)}"}
        else:
            yield {"type": "error", "data": f"Groq error: {str(e)}"}

def run_llm_with_fallback(llm_call_func, api_key: str, model_name: str) -> Any:
    """
    Executes a standard LLM function, retrying with FALLBACK_MODEL if a 429 rate limit is thrown.
    """
    try:
        llm = ChatGroq(model=model_name, groq_api_key=api_key, temperature=0.2)
        return llm_call_func(llm)
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "rate_limit" in err_msg.lower():
            logger.warning(f"Rate limited on {model_name} for generator task. Running fallback model {FALLBACK_MODEL}.")
            llm = ChatGroq(model=FALLBACK_MODEL, groq_api_key=api_key, temperature=0.2)
            return llm_call_func(llm)
        raise e

def generate_summary(
    repository_name: str,
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Generates summary, utilizing fallbacks on rate limits.
    """
    search_query = "README index main config setup requirements package config structure architecture overview"
    docs = robust_retrieve(repository_name, search_query, top_k=12)
    context = "\n".join([f"--- File Context {i+1} ---\n{d.page_content}" for i, d in enumerate(docs)])
    
    prompt = SUMMARY_PROMPT.format(repository_name=repository_name, context=context)
    
    try:
        return run_llm_with_fallback(lambda llm: llm.invoke(prompt).content, api_key, model_name)
    except Exception as e:
        logger.error(f"Groq summary generation failed: {e}")
        return f"Error generating summary: {e}"

def generate_interview_questions(
    repository_name: str,
    difficulty: str,
    count: int,
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Gathers key programmatic modules and builds conceptual interview Q&As of varying difficulties.
    Always prepends 5 default questions answered specifically in the context of the project.
    """
    search_query = "classes interface functions routes databases api authentication controllers implementation logic"
    docs = robust_retrieve(repository_name, search_query, top_k=10)
    context = "\n".join([f"--- Snippet {i+1} ---\n{d.page_content}" for i, d in enumerate(docs)])
    
    additional_count = max(0, count - 5)
    
    prompt = f"""You are a Technical Interviewer. Your task is to generate high-quality technical interview questions and answers based on the codebase '{repository_name}'.
The first 5 questions MUST be these exact default questions, answered specifically based on the provided codebase context and architecture of the project '{repository_name}':

1. Explain your project.
2. What was your role in the project? (Answer as a developer/architect who built this project)
3. Why did you choose this technology stack?
4. What challenges did you face and how did you solve them?
5. If 10,000 users use your application simultaneously, how would you handle it?

After these 5 default questions, generate exactly {additional_count} additional {difficulty} level technical interview questions and answers based on the specific architecture, design choices, logic, and implementations in this codebase.
- If 'Beginner': Focus on basic file structure, setup, language syntax used, main entrypoints, and what individual components do.
- If 'Intermediate': Focus on design patterns, data flows, integration details, database queries, endpoints, and error handling.
- If 'Advanced': Focus on system architecture, performance optimizations, security vulnerabilities, concurrency, scalability, and system trade-offs.

Format the output cleanly. For each question, provide:
- The question number and text.
- A detailed "Answer" explaining the concepts clearly in the context of the project.
- A "Reference File(s)" section listing paths to files in the codebase where relevant logic can be found (especially for the codebase-specific questions).

Codebase Context:
{context}
"""
    try:
        return run_llm_with_fallback(lambda llm: llm.invoke(prompt).content, api_key, model_name)
    except Exception as e:
        return f"Failed to generate questions: {e}"

def generate_documentation(
    repository_name: str,
    doc_type: str,
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Retrieves contextual elements and compiles specialized technical markdown documentation.
    """
    if doc_type == "README":
        search_query = "readme markdown installation command instructions usage overview setup setup requirements"
    elif doc_type == "API Documentation":
        search_query = "controllers api routers request response json endpoints headers methods parameter url backend"
    else:
        search_query = "components architecture db design schema relations layers service model repository packages files"
        
    docs = robust_retrieve(repository_name, search_query, top_k=12)
    context = "\n".join([f"--- Source Context {i+1} ---\n{d.page_content}" for i, d in enumerate(docs)])
    
    prompt = DOCS_PROMPT.format(
        repository_name=repository_name,
        doc_type=doc_type,
        context=context
    )
    
    try:
        return run_llm_with_fallback(lambda llm: llm.invoke(prompt).content, api_key, model_name)
    except Exception as e:
        return f"Failed to compile documentation: {e}"

def generate_mermaid_diagram(
    repository_name: str,
    diagram_type: str,
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Retrieves core structures and writes complete Mermaid.js graph layouts.
    """
    search_query = "classes fields interface functions schema routes database connections call workflow execution sequence"
    docs = robust_retrieve(repository_name, search_query, top_k=10)
    context = "\n".join([f"--- Blueprint Context {i+1} ---\n{d.page_content}" for i, d in enumerate(docs)])
    
    prompt = MERMAID_PROMPT.format(
        repository_name=repository_name,
        diagram_type=diagram_type,
        context=context
    )
    
    try:
        return run_llm_with_fallback(lambda llm: llm.invoke(prompt).content, api_key, model_name)
    except Exception as e:
        return f"Failed to generate diagram: {e}"

def explain_code_snippet(
    file_path: str,
    language: str,
    code: str,
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Provides line-by-line detailed technical explanations for the specified file content.
    """
    prompt = EXPLAINER_PROMPT.format(
        file_path=file_path,
        language=language,
        code=code
    )
    
    try:
        return run_llm_with_fallback(lambda llm: llm.invoke(prompt).content, api_key, model_name)
    except Exception as e:
        return f"Failed to generate code explanation: {e}"
