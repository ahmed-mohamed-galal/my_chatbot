import glob
import os
import re
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
CV_DIR = BASE_DIR / "cvs"
CHAT_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"

# IMPORTANT: replace the two ABOUT ME placeholders with text taken from your CV.
SCOPE = """
Name: Ahmed Mohamed Galal (in Arabic: أحمد محمد جلال)
ABOUT ME (English): <PUT 2-3 SENTENCES FROM YOUR CV HERE>
ABOUT ME (Arabic): <نفس الكلام بالعربي بصياغتك>
What I do: Data science, machine learning, deep learning, NLP, RAG and LLM chatbots.
Tools: Python, SQL, Scikit-learn, PyTorch, TensorFlow, LangChain, Ollama
What I do NOT do: mobile apps, game development, hardware/electronics, graphic design.
"""

OUT_OF_SCOPE_AR = "عذراً، هذا خارج نطاق خدماتي. أقدر أساعدك في الأسئلة عن خبراتي ومشاريعي."
OUT_OF_SCOPE_EN = (
    "Sorry, this is outside my services. "
    "I can help with questions about my skills, experience and projects."
)

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

PROMPT = ChatPromptTemplate(
    [
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
    ]
)


# ----------------------------------------------------------------------------
# Cached resources (built once, not on every Streamlit rerun)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading CV and building the index...")
def load_retriever():
    docs = []
    for path in glob.glob(str(CV_DIR / "*.pdf")):
        docs += PyMuPDFLoader(path).load()
    for path in glob.glob(str(CV_DIR / "*.md")):
        docs += TextLoader(path, encoding="utf-8").load()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    store = InMemoryVectorStore.from_documents(chunks, OllamaEmbeddings(model=EMBED_MODEL))
    return store.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 15})


@st.cache_resource
def load_llms():
    answer_llm = ChatOllama(model=CHAT_MODEL, temperature=0.2)
    judge_llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    return answer_llm, judge_llm


# ----------------------------------------------------------------------------
# Logic
# ----------------------------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(
        f"[{os.path.basename(d.metadata.get('source', ''))}]\n{d.page_content}" for d in docs
    )


def detect_lang(text: str) -> str:
    return "Arabic" if ARABIC_RE.search(text) else "English"


def refusal(question: str) -> str:
    return OUT_OF_SCOPE_AR if ARABIC_RE.search(question) else OUT_OF_SCOPE_EN


def in_scope(question: str, judge_llm) -> bool:
    if question.strip().lower().strip("!?.،؟ ") in GREETINGS:
        return True
    msg = f"""Decide if the question is about Ahmed Mohamed Galal: his skills, experience,
projects, education, certifications, contact info, or whether he does or does not offer a
certain service or technology. Questions in Arabic or English are both valid.

His scope:
{SCOPE}

ALWAYS answer YES for:
- Questions asking who Ahmed is or to introduce himself or tell about himself.
- Greetings and "what can you help me with" questions.
- "Do you do X?" or "Do you know X?" questions, even if the real answer is no.
General knowledge, jokes, or unrelated tasks are NO.

Examples:
"انت مين؟" -> YES
"عرفني عن نفسك" -> YES
"Tell me about yourself" -> YES
"Who are you?" -> YES
"What can you do?" -> YES
"Hello" -> YES
"مرحبا" -> YES
"ازاي اتواصل معاك؟" -> YES
"ايه خبرتك في Machine Learning؟" -> YES
"بتعمل تطبيقات موبايل؟" -> YES
"Do you know React?" -> YES
"ايه عاصمة فرنسا؟" -> NO
"What is the capital of France?" -> NO
"احكيلي نكتة" -> NO
"Write me a Python script" -> NO

Answer with exactly one word: YES or NO.
Question: {question}
Answer:"""
    out = (judge_llm.invoke(msg).content or "").strip().upper()
    return "YES" in out


def chat(question: str, retriever, answer_llm, judge_llm) -> str:
    if not in_scope(question, judge_llm):
        return refusal(question)

    lang = detect_lang(question)
    docs = retriever.invoke(question)
    msgs = PROMPT.invoke(
        {
            "context": SCOPE + "\n\n" + format_docs(docs),
            "question": question,
            "language": lang,
        }
    )
    answer = answer_llm.invoke(msgs).content

    # Safety net: English question must not get an Arabic answer
    if lang == "English" and ARABIC_RE.search(answer):
        answer = answer_llm.invoke(msgs).content
        if ARABIC_RE.search(answer):
            return "Sorry, something went wrong. Could you rephrase your question?"
    return answer


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Ahmed Galal | AI Assistant", page_icon="🤖")

# Let each paragraph pick its own direction (Arabic = RTL, English = LTR)
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
                reply = chat(question, retriever, answer_llm, judge_llm)
            except Exception as e:
                reply = f"Error talking to Ollama. Is it running? ({e})"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
