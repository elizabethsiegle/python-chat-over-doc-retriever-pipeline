import sys
import os
import logging
import PyPDF2
import docx  # Added import for handling .docx files

def extract_text(file_path, output_file):
    """Extract text from a PDF, DOCX, or TXT file and save it to a specified output file."""
    if not file_path:
        raise ValueError("No input file path provided")
    if not output_file:
        raise ValueError("No output file path provided")

    logging.info(f"Extracting text from {file_path}...")
    try:
        ext = os.path.splitext(file_path)[1].lower()
        text = ''
        if ext == '.pdf':
            with open(file_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    text += page.extract_text() or ''
        elif ext == '.docx':
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + '\n'
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as txt_file:
                text = txt_file.read()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write the extracted text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)

        if not os.path.exists(output_file):
            raise RuntimeError(f"Failed to create output file: {output_file}")

        return output_file

    except Exception as e:
        logging.error(f"Error extracting text: {str(e)}")
        raise

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python get_text.py <file_path> <output_text_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    output_file = sys.argv[2]
    extract_text(file_path, output_file)