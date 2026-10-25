import os
import re
from pathlib import Path

import dotenv
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Page config MUST be the first Streamlit command
st.set_page_config(page_title="Ahmed Galal | AI Assistant", page_icon="🤖")

# 2. Paths & Environment Setup
BASE_DIR = Path(__file__).resolve().parent
CV_DIR = BASE_DIR / "cvs"

dotenv.load_dotenv(BASE_DIR.parent / ".env")
dotenv.load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

# 3. Constants & Configuration
CHAT_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "openai/gpt-oss-20b"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_QUESTION_CHARS = 500

SCOPE = """
Name: Ahmed Mohamed Galal (in Arabic: أحمد محمد جلال)
ABOUT ME (English): <PUT 2-3 SENTENCES FROM YOUR CV HERE>
ABOUT ME (Arabic): <نفس الكلام بالعربي بصياغتك>
What I do: Data science, machine learning, deep learning, NLP, RAG and LLM chatbots.
Tools: Python, SQL, Scikit-learn, PyTorch, TensorFlow, LangChain, Ollama
What I do NOT do: mobile apps, game development, hardware/electronics, graphic design.
""".strip()

OUT_OF_SCOPE_AR = "عذراً، هذا خارج نطاق خدماتي. أقدر أساعدك في الأسئلة عن خبراتي ومشاريعي."
OUT_OF_SCOPE_EN = (
    "Sorry, this is outside my services. "
    "I can help with questions about my skills, experience and projects."
)
TOO_LONG_AR = "رسالتك طويلة جداً. من فضلك اختصرها."
TOO_LONG_EN = "Your message is too long. Please shorten it."

GREETING_REPLY_AR = "أهلاً بك! أنا المساعد الذكي لأحمد محمد جلال. كيف يمكنني مساعدتك اليوم بخصوص خبراتي ومشاريعي؟"
GREETING_REPLY_EN = "Hello! I'm Ahmed Mohamed Galal's AI assistant. How can I help you regarding my skills, experience, or projects?"

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
CLEAN_PUNCT_RE = re.compile(r"[!?.،؟\s]+$")

GREETINGS = {
    "hello", "hi", "hey", "good morning", "good evening",
    "مرحبا", "مرحباً", "اهلا", "أهلا", "السلام عليكم", "هاي", "صباح الخير", "مساء الخير",
}

# 4. Pre-compiled Prompts
JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Decide if the user's question is about Ahmed Mohamed Galal (his skills, experience, "
        "projects, education, certifications, contact info, or whether he offers a specific "
        "service/technology).\n\n"
        f"His scope:\n{SCOPE}\n\n"
        "Rules:\n"
        "- Answer YES for introductions, greetings, 'what can you do', or 'do you know/do X?' "
        "(even if X is not in his skills).\n"
        "- Answer NO for general knowledge, coding requests, jokes, or unrelated tasks.\n"
        "- Reply with ONLY one word: YES or NO."
    ),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "LANGUAGE RULE: You MUST reply ONLY in {language}. "
        "Do not mix in any other language. If {language} is English, write everything "
        "in English, including the name (Ahmed Mohamed Galal) and technical terms.\n"
        "For 'tell me about yourself' questions, use the ABOUT ME text in the requested "
        "language almost verbatim. Do not add anything to it.\n"
        "You are the professional AI assistant for Ahmed Mohamed Galal.\n"
        "The name Ahmed Mohamed Galal must be written in Arabic exactly as: أحمد محمد جلال "
        "(never غالال or any other spelling). In English write it as Ahmed Mohamed Galal.\n"
        "Your job is to answer questions about his CV, skills, experience, and what he can do.\n"
        "Answer ONLY using the provided context (his CV and scope notes). "
        "Keep names and numbers exactly as written.\n"
        "Never invent skills, experience, or prices that are not in the context.\n"
        "Do not add praise or adjectives (e.g. prestigious, strong, excellent) "
        "that are not in the context.\n"
        "Always speak in the first person as Ahmed (I, my), never in the third person.\n"
        "When replying in Arabic, use these terms: Machine Learning = تعلم الآلة, "
        "Data Scientist = عالم بيانات, Internship = تدريب.\n\n"
        "How to match the user's request with the context:\n"
        "- Match by MEANING, not only exact words. Treat abbreviations, synonyms and "
        "Arabic/English variants as the same thing "
        "(e.g. ML = Machine Learning, DL = Deep Learning, NLP = Natural Language "
        "Processing, sklearn = Scikit-learn, 'تعلم الآلة' = Machine Learning, "
        "'شات بوت' = chatbot).\n"
        "- A skill counts as known if it, or a closely related tool or technique, "
        "appears anywhere in the context (skills list, projects, certifications, "
        "education).\n"
        "- If the user asks about a broad area (e.g. 'data science', 'AI', "
        "'programming'), list the relevant skills from the context.\n"
        "- If the user asks what I can do or know, summarize my main skills and "
        "projects in a short organized answer.\n\n"
        "Rules for 'do you do/know X?' questions:\n"
        "- If X, or something closely related to X, appears in the context: answer yes "
        "and give the specific details (tools, projects, results).\n"
        "- If X is listed under 'What I do NOT do': say clearly it is not part of my "
        "services, in one friendly sentence, and mention one related thing I do offer.\n"
        "- If X is not mentioned anywhere: say it is not mentioned in my experience, "
        "in one friendly sentence, and mention one related thing I do offer.\n"
        "Do not use a generic refusal for these questions.\n\n"
        "Context:\n{context}",
    ),
    ("human", "Answer in {language} only.\n\nQuestion: {question}"),
])


# 5. Cached Resources
@st.cache_resource(show_spinner="Loading CV and building the index...")
def load_retriever():
    if not CV_DIR.exists():
        return None

    docs = []
    for path in CV_DIR.glob("*.pdf"):
        docs.extend(PyMuPDFLoader(str(path)).load())
    for path in CV_DIR.glob("*.md"):
        docs.extend(TextLoader(str(path), encoding="utf-8").load())

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    return store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12})


@st.cache_resource
def load_llms():
    answer_llm = ChatGroq(
        model=CHAT_MODEL,
        temperature=0.2,
        max_tokens=1500,
        reasoning_effort="low",
    )
    judge_llm = ChatGroq(
        model=JUDGE_MODEL,
        temperature=0,
        max_tokens=10,  # Reduced from 200: we only need "YES" or "NO"
        reasoning_effort="low",
    )
    return answer_llm, judge_llm


# 6. Helper Functions
def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{Path(d.metadata.get('source', '')).name}]\n{d.page_content}" for d in docs
    )


def is_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text))


def is_greeting(text: str) -> bool:
    cleaned = CLEAN_PUNCT_RE.sub("", text.strip().lower())
    return cleaned in GREETINGS


def in_scope(question: str, judge_llm) -> bool:
    chain = JUDGE_PROMPT | judge_llm
    out = (chain.invoke({"question": question}).content or "").strip().upper()
    return out.startswith("YES")


def generate_response(question: str, retriever, answer_llm, judge_llm) -> str:
    arabic_query = is_arabic(question)

    # Fast-path 1: Length check
    if len(question) > MAX_QUESTION_CHARS:
        return TOO_LONG_AR if arabic_query else TOO_LONG_EN

    # Fast-path 2: Instant greeting response (saves 2 LLM calls + vector search)
    if is_greeting(question):
        return GREETING_REPLY_AR if arabic_query else GREETING_REPLY_EN

    # Scope check
    if not in_scope(question, judge_llm):
        return OUT_OF_SCOPE_AR if arabic_query else OUT_OF_SCOPE_EN

    # Retrieval & Answering
    lang = "Arabic" if arabic_query else "English"
    docs = retriever.invoke(question)
    context_str = f"{SCOPE}\n\n{format_docs(docs)}"

    msgs = ANSWER_PROMPT.invoke(
        {"context": context_str, "question": question, "language": lang}
    )
    answer = answer_llm.invoke(msgs).content

    # Language enforcement retry with explicit correction
    if lang == "English" and is_arabic(answer):
        retry_msgs = msgs.to_messages() + [
            ("human", "Rewrite your previous response strictly in English only. Remove all Arabic script.")
        ]
        answer = answer_llm.invoke(retry_msgs).content
        if is_arabic(answer):
            return "Sorry, something went wrong. Could you rephrase your question?"

    return answer


# 7. UI Layout
st.markdown(
    """
    <style>
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        unicode-bidi: plaintext;
        text-align: start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Ahmed Galal | AI Assistant")
st.caption("اسألني عن خبراتي ومشاريعي · Ask me about my skills, experience and projects")

with st.sidebar:
    st.subheader("Settings")
    st.write(f"Model: `{CHAT_MODEL}`")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Reload CV files"):
        st.cache_resource.clear()
        st.rerun()

retriever = load_retriever()
answer_llm, judge_llm = load_llms()

if retriever is None:
    st.error(f"No PDF or .md files found in: {CV_DIR}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = generate_response(question, retriever, answer_llm, judge_llm)
            except Exception as e:
                reply = f"Error talking to the model. Please try again in a moment. ({e})"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
