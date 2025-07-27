# TechWriter - A tool to convert PDF documents into Medium articles using AI
import os
import time
from datetime import datetime
from pathlib import Path
import json
import hashlib
import math
import requests
import numpy as np
from tqdm.notebook import tqdm

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import gradio as gr


def format_time(seconds):
    """Convert seconds to human readable time format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

# Vector store configuration
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "vector_store")
PROGRESS_FILE = os.path.join(PERSIST_DIRECTORY, "processing_progress.json")

def verify_ollama_connection():
    """Verify that Ollama is running and accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("✓ Ollama connection verified")
            return True
        else:
            print("✗ Ollama is not responding correctly")
            return False
    except requests.exceptions.RequestException as e:
        print("✗ Could not connect to Ollama. Is it running?")
        print(f"Error: {str(e)}")
        return False

def test_embeddings():
    """Test that embeddings are working correctly."""
    try:
        embeddings = OllamaEmbeddings(model="llama3")
        test_text = "This is a test sentence."
        result = embeddings.embed_query(test_text)
        
        if isinstance(result, list) and len(result) > 0 and all(isinstance(x, float) for x in result):
            print("✓ Embeddings test successful")
            return True
        else:
            print("✗ Invalid embedding output format")
            return False
    except Exception as e:
        print("✗ Embeddings test failed")
        print(f"Error: {str(e)}")
        return False

def save_processing_progress(processed_files):
    """Save the list of processed files and their hashes."""
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(processed_files, f)

def load_processing_progress():
    """Load the list of previously processed files."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}

def get_file_hash(file_path):
    """Calculate hash of file for tracking changes."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def initialize_vector_store():
    """Initialize or load existing vector store."""
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    embeddings = OllamaEmbeddings(model="llama3")
    
    if os.path.exists(os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3")):
        print("Loading existing vector store...")
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    else:
        print("Creating new vector store...")
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

def save_vector_store(vectorstore):
    """Save vector store to disk."""
    print("Saving vector store...")
    vectorstore.persist()
    print(f"Vector store saved to {PERSIST_DIRECTORY}")

def clear_vector_store():
    """Clear the vector store and processing history."""
    import shutil
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        os.makedirs(PERSIST_DIRECTORY)
        print("Vector store cleared successfully")

def get_vector_store_stats():
    """Get statistics about the vector store."""
    if os.path.exists(os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3")):
        vectorstore = initialize_vector_store()
        collection = vectorstore._collection
        stats = {
            "total_documents": collection.count(),
            "persist_directory": PERSIST_DIRECTORY,
            "processed_files": len(load_processing_progress())
        }
        return stats
    return {
        "total_documents": 0, 
        "persist_directory": PERSIST_DIRECTORY, 
        "processed_files": 0
    }



article_template = """
You are Kaustav Chanda, an expert technical writer and software engineer creating a Medium article to showcase your skills in AI and ML. 
You are targeting to showcase your skills to potential employers and clients.
Topic: {topic}
Retrieved Content: {context}

Write a 1500-word technical article that follows Medium's style and markdown formatting:
1. Uses active voice throughout
2. Explains deep technical concepts in business-friendly language
3. Highlights practical applications and business value
4. Includes relevant examples from the source material
5. Structures content with clear headings and subheadings
6. Uses proper Markdown formatting with an author bio at the end

Article:
"""

article_prompt = PromptTemplate(
    input_variables=["topic", "context"],
    template=article_template
)

# Initialize Ollama with llama3 model
llm = Ollama(model="llama3")
article_chain = LLMChain(llm=llm, prompt=article_prompt)

def generate_article(topic, vectorstore):
    """Generate a Medium article based on the topic and vector store content."""
    # Search the vector store for relevant content
    results = vectorstore.similarity_search(topic, k=4)
    context = "\n".join([doc.page_content for doc in results])
    
    # Generate the article
    article = article_chain.run(topic=topic, context=context)
    return article

def gradio_interface(topic):
    """Gradio interface function for article generation."""
    try:        
        # Initialize vector store
        vectorstore = initialize_vector_store()
        if not vectorstore:
            return "Error: Could not initialize vector store. Please check if the store exists and contains documents."
        
        article = generate_article(topic, vectorstore)
        print(f"Generated article for topic: {topic}")
        print(f"Article length: {len(article.split())} words")
        return article
    except Exception as e:
        return f"Error generating article: {str(e)}"
    
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Enter the topic for your technical article...",
        label="Article Topic"
    ),
    outputs=gr.Markdown(
        label="Generated Article"
    ),
    title="Technical Article Generator",
    description="Generate a 1200-word technical article from your PDF content"
)

if __name__ == "__main__":
    if not verify_ollama_connection():
        print("Please ensure Ollama is running before starting the application.")
    elif not test_embeddings():
        print("Please check your Ollama embeddings configuration.")
    else:
        print("Starting Gradio interface...")
        iface.launch()