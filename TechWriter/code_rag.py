

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
# Setup the article generation chain
article_template = """
You are an expert technical writer and software engineer creating a Medium article to showcase your skills in AI and ML.
Topic: {topic}
Retrieved Content:
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

code_directory = "code"

# Initialize Ollama with llama3 model
llm = Ollama(model="llama3")
article_chain = LLMChain(llm=llm, prompt=article_prompt)

def generate_article(topic, code_vectorstore):
    """Generate a Medium article based on the topic and both vector stores' content."""
    code_results = code_vectorstore.similarity_search(topic, k=3)
    code_context = "\n".join([doc.page_content for doc in code_results])
    
    # Generate the article
    article = article_chain.run(topic=topic, code_context=code_context)
    # Save the article to a time-stamped file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"generated_article_{timestamp}.md", "w") as f:
        f.write(article)
    return article

def gradio_interface(topic):
    """Gradio interface function for article generation."""
    try:
        has_code =  any(Path(code_directory).rglob("*.*"))
        
        if not has_code:
            return "Error: No code files found. Please add some files first."
        
        code_vectorstore = initialize_code_store()
        
        if not code_vectorstore:
            return "Error: Could not initialize vector stores. Please check if the stores exist and contain documents."
        
        article = generate_article(topic, code_vectorstore)
        return article
    except Exception as e:
        return f"Error generating article: {str(e)}"


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



