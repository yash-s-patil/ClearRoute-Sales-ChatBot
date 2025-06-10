# ClearRoute Sales Chatbot

ClearRoute Sales Chatbot is an interactive web app that lets you upload multiple PDF files and chat with an AI assistant to extract information from your documents. It’s designed for sales teams, researchers, and anyone who wants to quickly find answers inside large or multiple PDFs.

## 🚀 Features

- Upload Multiple PDFs: Easily upload one or more PDF files at once.
- Conversational AI: Ask questions in natural language and get answers based only on your uploaded PDFs.
- Context-Restricted Answers: The chatbot only answers from your documents—if the answer isn’t found, it tells you.
- Source Tracking: See which PDF(s) the answer was found in.
- Secure: Your OpenAI API key is never stored.
- Fast Search: Uses vector search (FAISS) for quick document retrieval.

## 🛠️ Tech Stack

- Streamlit: Builds the interactive web UI.
- Python: Main programming language.
- OpenAI: Provides the GPT model for chat and embeddings.
- LangChain: Orchestrates document loading, splitting, and retrieval chains.
- FAISS: Efficient vector search for document chunks.
- PyPDF: Reads and parses PDF files.
- sentence-transformers: (Optional) For advanced embeddings if needed.

## 📦 Installation & Setup

1. Clone the Repository:

```
git clone https://github.com/yash-s-patil/ClearRoute-Sales-ChatBot.git
cd ClearRoute-Sales-ChatBot
```

2. Create a Virtual Environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Dependencies:
- Using requirements.txt:
```
pip3 install -r requirements.txt
```
- Or install manually:
```
pip3 install streamlit openai langchain faiss-cpu pypdf sentence-transformers
```

4. Run the App: 
```
streamlit run app.py
```

📝 Usage Instructions
1. Start the App:

After running `streamlit run app.py`, your browser will open the app.

2. Enter OpenAI API Key:

In the left sidebar, enter your OpenAI API key.

3. Upload PDFs:

Use the sidebar to upload one or more PDF files.

4. Chat:

Type your questions in the chat box. The chatbot will answer using only your PDFs. If the answer isn’t found, it will say: 

`Out of context. No relevant information found in the uploaded PDFs.`








