# 🤖 Ahmed Galal | AI CV Assistant

A bilingual (English / Arabic) chatbot that answers questions about my skills, experience and projects, using my CV as its only source of truth.

**🔗 Live demo:** [ahmed-galal-chatbot.streamlit.app]([https://ahmed-galal-chatbot.streamlit.app/](https://ahmed-galal-chatbot.streamlit.app/))

## ✨ Features

- **RAG over my CV**: PDF/Markdown files are split into chunks, embedded, and retrieved with MMR to build the answer context.
- **Bilingual**: replies in the same language as the question (Arabic or English), with proper RTL/LTR text handling in the UI.
- **Scope filter**: a small LLM "judge" decides whether a question is about me. Unrelated questions (general knowledge, jokes, etc.) get a polite refusal.
- **Grounded answers**: the prompt forces the model to use only the CV context, keep names and numbers exact, and never invent skills.
- **Language safety net**: if an English question gets an Arabic answer, the app retries before returning a message.
- **Input limit**: long messages are rejected to protect the free API quota.

## 🧱 Tech Stack

| Part | Tool |
|------|------|
| UI | Streamlit |
| Orchestration | LangChain |
| Chat model | `openai/gpt-oss-120b` via Groq |
| Scope judge | `openai/gpt-oss-20b` via Groq |
| Embeddings | FastEmbed (`paraphrase-multilingual-MiniLM-L12-v2`), runs locally |
| Vector store | LangChain `InMemoryVectorStore` |
| Document loading | PyMuPDF, TextLoader |

## 🏗️ How it works

```
Question → length check → scope judge (YES/NO)
                              │
                 NO → polite refusal
                 YES → detect language → retrieve top CV chunks (MMR)
                          → prompt + context → Groq LLM → answer
```

## 🚀 Run locally

```bash
git clone https://github.com/ahmed-mohamed-galal/my_chatbot.git
cd my_chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Put your CV (`.pdf` or `.md`) inside the `cvs/` folder.
2. Edit the `SCOPE` text in the main file (about me, skills, what you do / don't do).
3. Create `.streamlit/secrets.toml`:

```toml
   GROQ_API_KEY = "your-groq-key"
```

   Get a free key at https://console.groq.com/keys
4. Run:

```bash
   streamlit run chatbot.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push the repo to GitHub (never commit your API key).
2. Create a new app on https://share.streamlit.io and select the main file.
3. Add `GROQ_API_KEY` under **Advanced settings → Secrets**.

## 📁 Project structure

```
├── chatbot.py
├── requirements.txt
└── cvs/
    └── <your CV files>
```

## 📌 Notes

- Model names and free-tier limits on Groq change over time. Check the current list at https://console.groq.com/docs/models.
- The vector index is rebuilt in memory when the app starts.

## 📬 Contact

Ahmed Mohamed Galal · [LinkedIn](https://www.linkedin.com/in/ahmed-mohamed-galal/) · ahmed.mohamed.galal.2026@gmail.com
