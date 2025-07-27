# %% [markdown]
# # Code to Medium Article Generator with Persistent Vector Store
# 
# This notebook creates a system that:
# 1. Reads code files from a specified folder
# 2. Persistently stores their content in a Chroma vector store
# 3. Generates technical Medium articles through a Gradio chat interface
# 4. Allows resuming processing from previous state
# 5. Formats articles in active voice, targeting recruiters and managers
# 
# Key Features:
# - Persistent vector store using SQLite backend
# - Progress tracking and state management
# - Chunk-based processing with automatic saves
# - Resume capability for interrupted processing

# %%
# Install required packages
!pip install langchain langchain-community chromadb pypdf gradio requests numpy tqdm

# %%
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
from langchain.schema import Document
import gradio as gr

# %% [markdown]
# # Vector Store Configuration and Helper Functions
# 
# We'll set up:
# 1. Directory configuration for persistent storage
# 2. Helper functions for time formatting and progress tracking
# 3. Vector store initialization and management functions
# 4. Processing progress tracking and state management

# %%
# Helper function for time formatting
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
CODE_PERSIST_DIRECTORY = os.path.join(os.getcwd(), "code_store")
PROGRESS_FILE = os.path.join(PERSIST_DIRECTORY, "processing_progress.json")
CODE_PROGRESS_FILE = os.path.join(CODE_PERSIST_DIRECTORY, "code_progress.json")

# Supported code file extensions
CODE_EXTENSIONS = {
    'python': ['.py'],
    'javascript': ['.js', '.jsx', '.ts', '.tsx'],
    'java': ['.java'],
    'csharp': ['.cs']
}

# Helper function for time formatting
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

# %%
def initialize_code_store():
    """Initialize or load existing code vector store."""
    os.makedirs(CODE_PERSIST_DIRECTORY, exist_ok=True)
    embeddings = OllamaEmbeddings(model="llama3")
    
    if os.path.exists(os.path.join(CODE_PERSIST_DIRECTORY, "chroma.sqlite3")):
        print("Loading existing code vector store...")
        return Chroma(
            persist_directory=CODE_PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    else:
        print("Creating new code vector store...")
        return Chroma(
            persist_directory=CODE_PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

def save_code_progress(processed_files):
    """Save the list of processed code files and their hashes."""
    os.makedirs(CODE_PERSIST_DIRECTORY, exist_ok=True)
    with open(CODE_PROGRESS_FILE, "w") as f:
        json.dump(processed_files, f)

def load_code_progress():
    """Load the list of previously processed code files."""
    if os.path.exists(CODE_PROGRESS_FILE):
        with open(CODE_PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}

def clear_code_store():
    """Clear the code vector store and processing history."""
    import shutil
    if os.path.exists(CODE_PERSIST_DIRECTORY):
        shutil.rmtree(CODE_PERSIST_DIRECTORY)
        os.makedirs(CODE_PERSIST_DIRECTORY)
        print("Code vector store cleared successfully")

def get_code_store_stats():
    """Get statistics about the code vector store."""
    if os.path.exists(os.path.join(CODE_PERSIST_DIRECTORY, "chroma.sqlite3")):
        vectorstore = initialize_code_store()
        collection = vectorstore._collection
        stats = {
            "total_documents": collection.count(),
            "persist_directory": CODE_PERSIST_DIRECTORY,
            "processed_files": len(load_code_progress())
        }
        return stats
    return {
        "total_documents": 0, 
        "persist_directory": CODE_PERSIST_DIRECTORY, 
        "processed_files": 0
    }

# %% [markdown]
# # Document Processing and Vector Store Population
# 
# Now we'll implement the core document processing functionality:
# 1. Load PDFs from the specified directory
# 2. Split documents into chunks
# 3. Process and store embeddings with progress tracking
# 4. Save state at regular intervals

# %%
def load_code_files(directory):
    """Load all code files from the specified directory recursively."""
    code_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = Path(os.path.join(root, file))
            file_ext = file_path.suffix.lower()
            
            # Check if file extension is in supported extensions
            for lang_exts in CODE_EXTENSIONS.values():
                if file_ext in lang_exts:
                    code_files.append(file_path)
                    break
    
    if not code_files:
        print("No supported code files found in the specified directory.")
        return []
    
    print(f"\nLoading {len(code_files)} code files...")
    documents = []
    
    for file_path in code_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Create a document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': str(file_path),
                        'file_type': file_path.suffix.lower(),
                        'file_name': file_path.name
                    }
                )
                documents.append(doc)
                print(f"✓ Loaded {file_path.name}")
        except Exception as e:
            print(f"✗ Error loading {file_path.name}: {str(e)}")
    
    return documents

def process_code_files(documents, vectorstore=None):
    """Process code files and store in the code vector store."""
    if not documents:
        print("No code documents to process.")
        return initialize_code_store()
    
    # Verify Ollama connection and embeddings before processing
    if not verify_ollama_connection() or not test_embeddings():
        raise RuntimeError("Failed to initialize Ollama and embeddings. Please check the error messages above.")
    
    # Initialize or load vector store
    if vectorstore is None:
        vectorstore = initialize_code_store()
    
    # Load processing progress
    processed_files = load_code_progress()
    
    # Initialize timing and progress tracking
    start_time = time.time()
    total_docs = len(documents)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting processing of {total_docs} code files")
    
    # Text splitter optimized for code
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Larger chunks for code to maintain context
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "    ", "\t"]
    )
    
    # Process documents with progress tracking
    processed_chunks = 0
    save_interval = 20  # Save every 20 chunks
    last_saved_chunk = 0
    
    for i, doc in enumerate(documents, 1):
        # Check if document was already processed
        if doc.metadata.get('source') in processed_files:
            print(f"Skipping already processed file: {doc.metadata.get('source')}")
            continue
            
        print(f"\nProcessing file {i}/{total_docs}: {doc.metadata.get('source', 'Unknown Source')}")
        chunk_start = time.time()
        texts = text_splitter.split_documents([doc])
        processed_chunks += len(texts)
        
        try:
            # Add to vector store with retry logic
            vectorstore.add_documents(texts)
            
            # Save progress periodically
            if processed_chunks > last_saved_chunk + save_interval:
                vectorstore.persist()
                print(f"\nProgress saved. Total chunks processed: {processed_chunks}")
                last_saved_chunk = processed_chunks
            
            # Update progress tracking
            processed_files[doc.metadata.get('source')] = get_file_hash(doc.metadata.get('source'))
            save_code_progress(processed_files)
            
            # Calculate and show progress
            chunk_time = time.time() - chunk_start
            print(f"Chunks in this file: {len(texts)}")
            print(f"Time for this file: {format_time(chunk_time)}")
            
        except Exception as e:
            print(f"\nERROR processing file {i}: {str(e)}")
            print("Continuing with next file...")
            continue
    
    # Final save
    vectorstore.persist()
    save_code_progress(processed_files)
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing completed:")
    print(f"Total files processed: {total_docs}")
    print(f"Total chunks created: {processed_chunks}")
    print(f"Total processing time: {format_time(total_time)}")
    
    # Show vector store stats
    stats = get_code_store_stats()
    print("\nCode Vector Store Statistics:")
    print(f"Total documents in store: {stats['total_documents']}")
    print(f"Total processed files: {stats['processed_files']}")
    print(f"Store location: {stats['persist_directory']}")
    
    return vectorstore

# %%
def load_pdfs(directory):
    """Load all PDFs from the specified directory."""
    pdf_files = list(Path(directory).glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the specified directory.")
        return []
    
    print(f"\nLoading {len(pdf_files)} PDF files...")
    documents = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())
            print(f"✓ Loaded {pdf_path.name}")
        except Exception as e:
            print(f"✗ Error loading {pdf_path.name}: {str(e)}")
    return documents

def process_documents(documents, vectorstore=None):
    """Split documents and create/update vector store with progress tracking."""
    if not documents:
        print("No documents to process.")
        return initialize_vector_store()
    
    # Verify Ollama connection and embeddings before processing
    if not verify_ollama_connection() or not test_embeddings():
        raise RuntimeError("Failed to initialize Ollama and embeddings. Please check the error messages above.")
    
    # Initialize or load vector store
    if vectorstore is None:
        vectorstore = initialize_vector_store()
    
    # Load processing progress
    processed_files = load_processing_progress()
    
    # Initialize timing and progress tracking
    start_time = time.time()
    total_docs = len(documents)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting processing of {total_docs} documents")
    
    # Text splitter with optimized chunk size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    # Estimate chunks and time
    sample_size = min(5, len(documents))
    sample_chunks = sum(len(text_splitter.split_text(doc.page_content)) for doc in documents[:sample_size])
    estimated_total_chunks = math.ceil((sample_chunks / sample_size) * len(documents))
    print(f"Estimated total chunks: {estimated_total_chunks} (based on {sample_size} document sample)")
    
    # Process documents with progress tracking
    processed_chunks = 0
    save_interval = 30  # Save every 30 chunks for better persistence
    last_saved_chunk = 0
    
    for i, doc in enumerate(documents, 1):
        # Check if document was already processed
        if doc.metadata.get('source') in processed_files:
            print(f"Skipping already processed document: {doc.metadata.get('source')}")
            continue
        print(f"\nProcessing document {i}/{total_docs}: {doc.metadata.get('source', 'Unknown Source')}")
        chunk_start = time.time()
        texts = text_splitter.split_documents([doc])
        processed_chunks += len(texts)
        
        try:
            # Add to vector store with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vectorstore.add_documents(texts)
                    
                    # Save progress periodically
                    if processed_chunks > last_saved_chunk + save_interval:
                        save_vector_store(vectorstore)
                        print(f"\nProgress saved. Total chunks processed: {processed_chunks}")
                        last_saved_chunk = processed_chunks
                    
                    break
                except ValueError as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Retry {attempt + 1}/{max_retries} - Waiting 2 seconds before retry...")
                    time.sleep(2)
        
            # Calculate progress and estimates
            chunk_time = time.time() - chunk_start
            elapsed_total = time.time() - start_time
            avg_time_per_chunk = elapsed_total / processed_chunks
            estimated_remaining_chunks = estimated_total_chunks - processed_chunks
            estimated_remaining_time = estimated_remaining_chunks * avg_time_per_chunk
            
            print(f"\nProgress: Document {i}/{total_docs}")
            print(f"Chunks in this document: {len(texts)}")
            print(f"Total chunks processed: {processed_chunks}/{estimated_total_chunks}")
            print(f"Time for this document: {format_time(chunk_time)}")
            print(f"Estimated remaining time: {format_time(estimated_remaining_time)}")
        
        except Exception as e:
            print(f"\nERROR processing document {i}: {str(e)}")
            print("Continuing with next document...")
            continue
    
    # Final save
    save_vector_store(vectorstore)
    processed_files[doc.metadata.get('source')] = get_file_hash(doc.metadata.get('source'))
    save_processing_progress(processed_files)
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing completed:")
    print(f"Total documents processed: {total_docs}")
    print(f"Total chunks created: {processed_chunks}")
    print(f"Total processing time: {format_time(total_time)}")
    print(f"Average time per chunk: {(total_time/processed_chunks):.2f} seconds")
    
    # Show vector store stats
    stats = get_vector_store_stats()
    print("\nVector Store Statistics:")
    print(f"Total documents in store: {stats['total_documents']}")
    print(f"Total processed files: {stats['processed_files']}")
    print(f"Store location: {stats['persist_directory']}")
    
    return vectorstore

# %% [markdown]
# # Article Generation with LLM Chain
# 
# We'll set up the article generation pipeline:
# 1. Create a prompt template for technical articles
# 2. Initialize the LLM chain with Ollama
# 3. Create a function to generate articles from vector store content
# 4. Add a Gradio interface for easy interaction

# %%
# Setup the article generation chain
article_template = """
You are an expert technical writer and software engineer creating a Medium article to showcase your skills in AI and ML.
Topic: {topic}
Retrieved Content:
PDF Content: {pdf_context}
Code Content: {code_context}

Write a 1200-word technical article that:
1. Uses active voice throughout
2. Explains technical concepts in business-friendly language
3. Highlights practical applications and business value
4. Includes relevant examples from both the documentation and code
5. Structures content with clear headings and subheadings

Article:
"""

article_prompt = PromptTemplate(
    input_variables=["topic", "pdf_context", "code_context"],
    template=article_template
)

pdf_directory = "pdfs123"
code_directory = "code"

# Initialize Ollama with llama3 model
llm = Ollama(model="llama3")
article_chain = LLMChain(llm=llm, prompt=article_prompt)

def generate_article(topic, pdf_vectorstore, code_vectorstore):
    """Generate a Medium article based on the topic and both vector stores' content."""
    # Search both vector stores for relevant content
    pdf_results = pdf_vectorstore.similarity_search(topic, k=3)
    code_results = code_vectorstore.similarity_search(topic, k=3)
    
    pdf_context = "\n".join([doc.page_content for doc in pdf_results])
    code_context = "\n".join([doc.page_content for doc in code_results])
    
    # Generate the article
    article = article_chain.run(topic=topic, pdf_context=pdf_context, code_context=code_context)
    return article

def gradio_interface(topic):
    """Gradio interface function for article generation."""
    try:
        has_pdfs = any(Path(pdf_directory).glob("*.pdf"))
        has_code =  any(Path(code_directory).rglob("*.*"))
        
        if not has_pdfs and not has_code:
            return "Error: No PDF files or code files found. Please add some files first."
        
        # Initialize vector stores
        pdf_vectorstore = initialize_vector_store()
        code_vectorstore = initialize_code_store()
        
        if not pdf_vectorstore or not code_vectorstore:
            return "Error: Could not initialize vector stores. Please check if the stores exist and contain documents."
        
        article = generate_article(topic, pdf_vectorstore, code_vectorstore)
        return article
    except Exception as e:
        return f"Error generating article: {str(e)}"

# %%
# Directory setup and initial processing
pdf_directory = "pdfs123"
code_directory = "code"
os.makedirs(pdf_directory, exist_ok=True)
os.makedirs(code_directory, exist_ok=True)

# Process files if they exist
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting file processing pipeline")

# Verify Ollama setup first
if verify_ollama_connection() and test_embeddings():
    # Show existing vector store stats if any
    print("\nCurrent Vector Store Status:")
    pdf_stats = get_vector_store_stats()
    code_stats = get_code_store_stats()
    print("\nPDF Vector Store:")
    print(f"Total documents: {pdf_stats['total_documents']}")
    print(f"Total processed files: {pdf_stats['processed_files']}")
    print("\nCode Vector Store:")
    print(f"Total documents: {code_stats['total_documents']}")
    print(f"Total processed files: {code_stats['processed_files']}")
    
    # Process PDFs
    if any(Path(pdf_directory).glob("*.pdf")):
        print("\nProcessing PDF files...")
        documents = load_pdfs(pdf_directory)
        vectorstore = process_documents(documents)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] PDF vector store creation completed")
    else:
        print("\nNo PDF files found in the 'pdfs' directory.")
    
    # Process code files
    if any(Path(code_directory).rglob("*.*")):
        print("\nProcessing code files...")
        code_documents = load_code_files(code_directory)
        code_vectorstore = process_code_files(code_documents)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Code vector store creation completed")
    else:
        print("\nNo code files found in the 'code' directory.")
else:
    print("Setup verification failed. Please check the error messages above.")

# %%
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
    description="Generate a 1000-word technical article from your PDF content, targeted at recruiters and managers."
)

# Launch the interface
iface.launch(share=True)

# %%
iface.close()  # Close the interface when done

# %% [markdown]
# # How to Use
# 
# 1. Create two directories in the same location as this notebook:
#    - `pdfs` for PDF documentation files
#    - `code` for source code files
# 2. Place your files in the appropriate directories:
#    - PDF files in the `pdfs` directory
#    - Code files (.py, .js, .java, .cs, etc.) in the `code` directory
# 3. Install Ollama on your system and start the Ollama service
# 4. Pull the llama3 model using: `ollama pull llama3`
# 5. Run all cells in this notebook
# 6. Use the Gradio interface to:
#    - Enter your desired article topic
#    - Click submit to generate the article
#    - Copy the generated article and paste it into Medium
# 
# The system will:
# - Automatically load and process both PDF and code files
# - Store embeddings persistently in separate stores:
#   - `vector_store` for PDFs
#   - `code_store` for code files
# - Save progress periodically during processing
# - Allow resuming from previous state if interrupted
# - Generate articles using knowledge from both documentation and code
# 
# Supported Code Files:
# - Python (.py)
# - JavaScript (.js, .jsx, .ts, .tsx)
# - Java (.java)
# - C# (.cs)
# 
# Notes:
# - Make sure Ollama is running before using this notebook
# - Both vector stores are persistent and will be reused across sessions
# - You can clear the stores using:
#   - `clear_vector_store()` for PDFs
#   - `clear_code_store()` for code files
# - Progress tracking prevents reprocessing of already processed files
# 
# Example Usage:
# ```python
# # Get current store statistics
# pdf_stats = get_vector_store_stats()
# code_stats = get_code_store_stats()
# print(f"PDFs in store: {pdf_stats['total_documents']}")
# print(f"Code files in store: {code_stats['total_documents']}")
# 
# # Clear stores and start fresh
# clear_vector_store()
# clear_code_store()
# 
# # Process new documents
# pdf_documents = load_pdfs(pdf_directory)
# code_documents = load_code_files(code_directory)
# 
# vectorstore = process_documents(pdf_documents)
# code_vectorstore = process_code_files(code_documents)
# 
# # Save explicitly if needed
# save_vector_store(vectorstore)
# code_vectorstore.persist()
# ```


