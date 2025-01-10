import sys
import json
import faiss
import numpy as np
import os
import logging
import pickle
from rank_bm25 import BM25Okapi # BM25 is a ranking function used in information retrieval
# It helps find relevant text chunks based on term frequency
# We combine it with FAISS (vector similarity) for better results
# FAISS finds semantic similarity, BM25 finds keyword matches
# Together they provide more accurate search results

# Example:
# FAISS might find: "The weather in San Francisco is mild year-round"
# when searching for: "What's the climate like in SF?"
# 
# BM25 might find: "SF weather stays between 60-70°F"
# when searching for: "San Francisco weather"
#
# Combining both gives us better results than either alone
from voyageai import Client
from clients import VOYAGE_API_KEY  # Import the key directly instead of Config class

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Voyage client
voyage_client = Client(api_key=VOYAGE_API_KEY)

def embed_texts(texts):
    """Embed a list of texts using the Voyage AI client."""
    response = voyage_client.embed(
        texts=texts,
        model="voyage-3",
        input_type="document"
    )
    return np.array(response.embeddings).astype('float32')

def create_index(chunks_file, output_dir='output'):
    """Create FAISS and BM25 indexes from chunks file."""
    logging.info(f"Creating indexes from {chunks_file}")
    
    try:
        # Verify input file exists
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

        # Read chunks
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        texts = chunks_data

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create FAISS index
        logging.info("Creating embeddings...")
        embeddings = embed_texts(texts)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Create BM25 index
        logging.info("Creating BM25 index...")
        tokenized_texts = [text.split() for text in texts]
        bm25 = BM25Okapi(tokenized_texts)

        # Save indexes and texts
        logging.info("Saving indexes and texts...")
        index_path = os.path.join(output_dir, 'faiss_index.index')
        texts_path = os.path.join(output_dir, 'texts.json')
        bm25_path = os.path.join(output_dir, 'bm25_index.pkl')

        faiss.write_index(index, index_path)
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False)
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25, f)

        # Verify files were created
        if not all(os.path.exists(p) for p in [index_path, texts_path, bm25_path]):
            raise RuntimeError("Failed to create one or more index files")

        logging.info("Index creation complete.")
        return True

    except Exception as e:
        logging.error(f"Error creating index: {str(e)}")
        raise

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python create_index.py <chunks_file>")
        sys.exit(1)
    
    chunks_file = sys.argv[1]
    create_index(chunks_file)

required_files = [
    'output/article_extracted.txt',
    'output/chunks.json',
    'output/faiss_index.index',
    'output/texts.json'
]

for file in required_files:
    if os.path.exists(file):
        print(f"{file} exists")
    else:
        print(f"{file} is missing")