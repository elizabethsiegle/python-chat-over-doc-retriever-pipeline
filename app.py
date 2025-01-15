import sys
import os
import subprocess
import json
import faiss
import numpy as np
import voyageai
import pickle # Python's built-in module for serializing objects
              # We use it to save/load the BM25 index to/from disk
              # Like JSON, but for Python objects that JSON can't handle
import requests
import tiktoken
import logging
from flask import Flask, render_template, request, jsonify
from urllib.parse import urlparse
import re
import trafilatura
from get_text import extract_text
from chunk_x_context import process_text
from create_index import create_index
import boto3
from botocore.client import Config
from clients import (
    VOYAGE_API_KEY,
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_AUTH_TOKEN
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Voyage client with API key
try:
    logger.debug(f"VOYAGE_API_KEY present: {bool(VOYAGE_API_KEY)}")
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    logger.debug("Successfully initialized Voyage client")
except Exception as e:
    logger.error(f"Failed to initialize Voyage client: {e}")
    raise

# Initialize tokenizer
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Define maximum tokens
MAX_TOTAL_TOKENS = 2097152  # Updated to allow for more detailed responses
MAX_RESPONSE_TOKENS = 2000000  # Increased to allow for longer answers
MAX_PROMPT_TOKENS = MAX_TOTAL_TOKENS - MAX_RESPONSE_TOKENS

# Global variables for index and texts
index = None
texts = None
bm25 = None

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Create directories for storing files
os.makedirs('downloads', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Add these configurations
UPLOAD_FOLDER = 'temp/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def download_pdf(url):
    """Download PDF from URL and save it locally."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract filename from URL or use default
        filename = os.path.basename(urlparse(url).path)
        if not filename.endswith('.pdf'):
            filename = 'downloaded.pdf'
            
        filepath = os.path.join('downloads', filename)
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        return filepath
    except Exception as e:
        raise Exception(f"Failed to download PDF: {str(e)}")

def process_pdf(file_path):
    """Process a PDF or DOCX file by extracting text, chunking, and building an index."""
    print(f"Processing {file_path}...")

    # Sanitize filename to replace spaces and special characters
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    extracted_text_file = os.path.join('output', 'article_extracted.txt')
    chunks_file = os.path.join('output', 'chunks.json')

    # Ensure the 'output' directory exists
    os.makedirs('output', exist_ok=True)

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Extract text from the file
        print(f"Extracting text from {file_path}...")
        subprocess.run(['python', os.path.join(script_dir, 'get_text.py'), file_path, extracted_text_file], check=True)

        # Chunk and contextualize the text
        print("Chunking and contextualizing the text...")
        subprocess.run(['python', os.path.join(script_dir, 'chunk_x_context.py'), extracted_text_file, chunks_file], check=True)

        # Build the index
        print("Building the index...")
        subprocess.run(['python', os.path.join(script_dir, 'create_index.py'), chunks_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during processing: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def embed_query(query):
    """Embed a query using the Voyage AI client."""
    response = voyage_client.embed(
        texts=[query],
        model="voyage-3",
        input_type="query"
    )
    return np.array(response.embeddings[0]).astype('float32')


def embed_texts(texts):
    """Embed a list of texts using the Voyage AI client."""
    response = voyage_client.embed(
        texts=texts,
        model="voyage-3",
        input_type="document"
    )
    return np.array(response.embeddings).astype('float32')


def clean_text(text):
    """Clean text of special characters and encoding issues."""
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove special tokens
    text = re.sub(r'<\|.*?\|>', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    return text.strip()


def retrieve_chunks(query, index, texts, bm25, k=5):
    """Retrieve relevant chunks using hybrid search (FAISS + BM25)"""
    try:
        # Get query embedding
        query_embedding = voyage_client.embed(
            texts=[query],
            model="voyage-3",
            input_type="query"
        ).embeddings[0]

        # FAISS search
        D, I = index.search(np.array([query_embedding]).astype('float32'), k)
        faiss_results = [(i, D[0][j]) for j, i in enumerate(I[0])]

        # BM25 search
        bm25_scores = bm25.get_scores(query.split())
        bm25_results = [(i, score) for i, score in enumerate(bm25_scores)]
        bm25_results.sort(key=lambda x: x[1], reverse=True)
        bm25_results = bm25_results[:k]

        # Combine results (simple averaging of normalized scores)
        combined_results = {}
        
        # Normalize and combine scores
        max_faiss_score = max(d for _, d in faiss_results)
        max_bm25_score = max(s for _, s in bm25_results)
        
        for idx, faiss_score in faiss_results:
            norm_faiss_score = faiss_score / max_faiss_score
            combined_results[idx] = norm_faiss_score

        for idx, bm25_score in bm25_results:
            norm_bm25_score = bm25_score / max_bm25_score
            combined_results[idx] = combined_results.get(idx, 0) + norm_bm25_score

        # Sort by combined score
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        
        # Return top chunks
        return [texts[idx] for idx, _ in sorted_results[:k]]

    except Exception as e:
        logger.error(f"Error in retrieve_chunks: {str(e)}")
        return []


def generate_answer(query, chunks):
    """Generate an answer to the query using Cloudflare's Llama-3.2."""
    if not chunks:
        return "I could not find relevant information in the document."

    # Combine chunks into context
    context = " ".join(chunks)
    
    try:
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant. Answer the question using ONLY the information provided in the context. If you cannot find the answer in the context, say 'I cannot find that information in the document.'"
            },
            {
                "role": "user", 
                "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            }
        ]
        
        response = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.2-1b-instruct",
            headers={"Authorization": f"Bearer {CLOUDFLARE_AUTH_TOKEN}"},
            json={"messages": messages}
        )
        
        if response.status_code != 200:
            return "An error occurred while generating the answer."

        result = response.json()
        if "result" in result and "response" in result["result"]:
            return result["result"]["response"].strip()
        else:
            return "An error occurred while processing the response."
            
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return "An error occurred while generating the answer."


def main(input_path):
    global index, texts, bm25

    if os.path.isfile(input_path) and input_path.lower().endswith(('.pdf', '.docx')):
        # Process single PDF or DOCX file
        process_pdf(input_path)
    elif os.path.isdir(input_path):
        # Process all PDF and DOCX files in the directory
        for root_dir, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(('.pdf', '.docx')):
                    file_path = os.path.join(root_dir, file)
                    process_pdf(file_path)
    else:
        print(f"Invalid path or no PDF or DOCX files found at: {input_path}")
        sys.exit(1)

    # Load indexes and texts
    try:
        # vector similarity search index
        index = faiss.read_index(os.path.join('output', 'faiss_index.index'))
        with open(os.path.join('output', 'texts.json'), 'r', encoding='utf-8') as f:
            texts = json.load(f)
        with open(os.path.join('output', 'bm25_index.pkl'), 'rb') as f:
            bm25 = pickle.load(f)
    except Exception as e:
        print(f"Error loading indexes and texts: {e}")
        sys.exit(1)


def extract_article_text(url):
    """Extract text from a news article URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # Try with requests first
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Try parsing with trafilatura
        text = trafilatura.extract(response.text, include_formatting=False)
        if text:
            return text.strip()
            
        # If trafilatura fails, try basic extraction
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_formatting=False)
            if text:
                return text.strip()
            
        logging.warning(f"Could not extract text from {url}")
        return None
        
    except Exception as e:
        logging.error(f"Error extracting article text: {e}")
        return None


@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST', 'OPTIONS'])
def process_url():
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        global index, texts, bm25
        
        # Get and validate URL
        url = request.get_json()
        if not url or 'url' not in url:
            return jsonify({'error': 'No URL provided'}), 400

        url = url['url']
        logger.debug(f"Processing URL: {url}")

        # Create temporary directories
        os.makedirs('temp/output', exist_ok=True)
        os.makedirs('temp/downloads', exist_ok=True)

        # Process the document locally
        if url.lower().endswith('.pdf'):
            pdf_path = download_pdf(url)
            process_pdf(pdf_path)
        else:
            article_text = extract_article_text(url)
            if not article_text:
                return jsonify({'error': 'Could not extract article text'}), 400
                
            text_path = os.path.join('temp/downloads', 'article.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(article_text)
                
            process_document(text_path)

        # Load the processed files for immediate use
        try:
            index_path = os.path.join('temp/output', 'faiss_index.index')
            texts_path = os.path.join('temp/output', 'texts.json')
            bm25_path = os.path.join('temp/output', 'bm25_index.pkl')
            
            index = faiss.read_index(index_path)
            with open(texts_path, 'r', encoding='utf-8') as f:
                texts = json.load(f)
            with open(bm25_path, 'rb') as f:
                bm25 = pickle.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load index or texts: {str(e)}")
            return jsonify({'error': 'Failed to load processed files'}), 500

        return jsonify({
            'success': True,
            'message': 'Document processed successfully'
        })

    except Exception as e:
        logger.error(f"Error in process_url: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        global index, texts, bm25
        
        # Check if we need to load the indexes
        if index is None or texts is None or bm25 is None:
            try:
                index_path = os.path.join('temp/output', 'faiss_index.index')
                texts_path = os.path.join('temp/output', 'texts.json')
                bm25_path = os.path.join('temp/output', 'bm25_index.pkl')
                
                index = faiss.read_index(index_path)
                with open(texts_path, 'r', encoding='utf-8') as f:
                    texts = json.load(f)
                with open(bm25_path, 'rb') as f:
                    bm25 = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load indexes: {str(e)}")
                return jsonify({'error': 'Please process a document first'}), 400

        query = request.json.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        chunks = retrieve_chunks(query, index, texts, bm25, k=20)
        if not chunks:
            return jsonify({'answer': 'No relevant information found in the document.'})
            
        answer = generate_answer(query, chunks)
        return jsonify({'answer': answer})
        
    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_document(input_file):
    """Process a document through the entire pipeline."""
    try:
        # Create temp directory
        os.makedirs('temp/output', exist_ok=True)
        
        # Step 1: Extract text
        extracted_file = os.path.join('temp/output', 'article_extracted.txt')
        extract_text(input_file, extracted_file)
        logging.info(f"Text extracted to {extracted_file}")

        # Step 2: Create chunks
        chunks_file = os.path.join('temp/output', 'chunks.json')
        process_text(extracted_file, chunks_file)
        logging.info(f"Chunks created at {chunks_file}")

        # Step 3: Create index
        create_index(chunks_file, 'temp/output')
        logging.info("Index created successfully")

        # Verify all required files exist
        required_files = [
            os.path.join('temp/output', 'article_extracted.txt'),
            os.path.join('temp/output', 'chunks.json'),
            os.path.join('temp/output', 'faiss_index.index'),
            os.path.join('temp/output', 'texts.json')
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")

        return True

    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        return False

# You can call this function from your route handlers
# For example:
# @app.route('/process', methods=['POST'])
# def process():
#     result = process_document('path/to/your/file')
#     return jsonify({'success': result})

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
            
        # Secure the filename and save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the uploaded file
        try:
            # Extract text
            text_path = os.path.join('temp/output', 'extracted.txt')
            extract_text(filepath, text_path)
            
            # Process the text (chunk and index)
            process_document(text_path)
            
            return jsonify({
                'success': True,
                'message': 'File processed successfully'
            })
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('output', exist_ok=True)
    os.makedirs('downloads', exist_ok=True)
    # Start the Flask app
    app.run(debug=True, port=5000)