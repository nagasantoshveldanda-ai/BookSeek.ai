# BookSeek.ai
Rag on chatbot as Bookseek.ai
BookSeek.AI – AI-Powered Study Assistant

BookSeek.AI is an interactive Streamlit-based application that helps users learn more effectively by allowing them to upload study materials (PDFs, notes, etc.) and query them using AI-powered question answering. The app uses Retrieval-Augmented Generation (RAG) to search across uploaded documents and generate context-aware answers.

Features

📚 PDF Upload & Processing – Upload multiple PDF files to build a searchable knowledge base.

⚡ Vector Database – Uses ChromaDB to store and retrieve document chunks efficiently.

🤖 AI-Powered Q&A – Leverages OpenRouter LLM for generating accurate and contextual answers.

💾 Persistent Storage – Automatically saves and reloads processed document embeddings.

🖌 Custom UI/UX – Enhanced chat interface with styled user and assistant messages, background images, and responsive layout.

🔍 Source References – Option to view source document excerpts used for each AI answer.

💬 Conversation History – Keeps track of multiple conversations and their Q&A pairs.

Tech Stack

Frontend/UI: Streamlit

Backend: Python

Vector Database: ChromaDB

Document Processing: LangChain document loaders & splitters

LLM Provider: OpenRouter API

Installation & Setup

Clone the repository

git clone https://github.com/yourusername/bookseek-ai.git
cd bookseek-ai


Create a virtual environment

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Set up environment variables
Create a .env file in the project root:

OPENROUTER_API_KEY=your_openrouter_api_key_here


Run the app

streamlit run app.py

Usage

Upload your study materials in PDF format from the sidebar.

Wait for the documents to be processed and stored in the vector database.

Ask questions in the chat interface to receive AI-powered answers.

Optionally, expand the "📖 View Sources" section to see references from your documents.
