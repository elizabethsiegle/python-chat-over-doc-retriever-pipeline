import sys
import json
import logging
import os

def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Adjust chunk to end at a sentence or paragraph
        if end < text_len:
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point != -1:
                end = start + break_point + 1
                chunk = text[start:end]
        
        chunks.append(chunk)
        start = end - overlap
        
    return chunks

def process_text(input_file, output_file):
    """Process text file into chunks and save them to JSON."""
    if not input_file:
        raise ValueError("No input file path provided")
    if not output_file:
        raise ValueError("No output file path provided")

    logging.info(f"Processing text from {input_file}...")
    
    try:
        # Ensure input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create chunks
        chunks = chunk_text(text)
        
        # Save chunks to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # Verify file was created
        if not os.path.exists(output_file):
            raise RuntimeError(f"Failed to create output file: {output_file}")
            
        return output_file
        
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        raise

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python chunk_x_context.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_text(input_file, output_file)