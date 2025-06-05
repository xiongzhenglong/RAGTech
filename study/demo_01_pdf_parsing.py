# study/demo_01_pdf_parsing.py
#
# NOTE: This script has significant external dependencies:
# 1. Docling models: Requires downloading specific models via `Pipeline.download_docling_models()`.
#    These models are necessary for the core parsing functionality.
# 2. Detectron2: The `LayoutDetector` component (and thus `DocumentParser`)
#    depends on Detectron2, which can be complex to install and may have GPU requirements.
# 3. Standard Python libraries: pdfminer.six, PyMuPDF (fitz), Pillow.
#
# Running this script successfully requires a pre-configured environment
# where these dependencies are met.

import sys
import os
from pathlib import Path

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_parsing import PDFParser
from src.pipeline import Pipeline

def main():
    # Ensure Docling models are downloaded
    # This might take a while if the models are not already downloaded
    print("Checking and downloading Docling models if necessary...")
    Pipeline.download_docling_models()
    print("Docling models are ready.")

    # Define the path to the sample PDF file
    # Using a relative path from the root of the project
    sample_pdf_path = "data/test_set/pdf_reports/194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf"
    print(f"Using sample PDF: {sample_pdf_path}")

    # Check if the sample PDF file exists
    if not os.path.exists(sample_pdf_path):
        print(f"Error: Sample PDF file not found at {sample_pdf_path}")
        print("Please ensure the data is available in the 'data/test_set/pdf_reports/' directory.")
        # As a fallback, let's try to list files in the directory to help debug
        print(f"Looking for data in: {os.path.abspath('data/test_set/pdf_reports/')}")
        if os.path.exists('data/test_set/pdf_reports/'):
             print(f"Files in 'data/test_set/pdf_reports/': {os.listdir('data/test_set/pdf_reports/')[:5]}") # Show first 5
        else:
            print("'data/test_set/pdf_reports/' directory does not exist.")
        return

    # Initialize PDF parser
    print("Initializing PDFParser...")
    try:
        # Create output directory for parsed results
        output_dir = Path("study/parsed_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PDFParser with output directory
        parser = PDFParser(output_dir=output_dir)
        
        # Parse the PDF document
        print(f"Parsing PDF: {sample_pdf_path}")
        pdf_path = Path(sample_pdf_path)
        
        # Parse and export the PDF
        parser.parse_and_export(input_doc_paths=[pdf_path])
        
        # Load and display the parsed JSON output
        output_json_path = output_dir / f"{pdf_path.stem}.json"
        if output_json_path.exists():
            import json
            with open(output_json_path, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)
            
            print(f"\nSuccessfully parsed document and saved to: {output_json_path}")
            
            # Display basic information about the parsed document
            if 'metainfo' in parsed_data:
                print(f"Document pages: {parsed_data['metainfo'].get('total_pages', 'Unknown')}")
            
            # Display first page content if available
            if 'content' in parsed_data and parsed_data['content']:
                first_page = parsed_data['content'][0]
                print(f"\n--- First Page Content Preview ---")
                if 'content' in first_page:
                    # Extract text from first few blocks
                    text_snippets = []
                    for block in first_page['content'][:3]:  # First 3 blocks
                        if 'text' in block and block['text']:
                            text_snippets.append(str(block['text']))
                    
                    if text_snippets:
                        preview_text = " ".join(text_snippets)
                        print(preview_text[:500] + "..." if len(preview_text) > 500 else preview_text)
                    else:
                        print("No text content found in the first page blocks.")
                else:
                    print("No content structure found in the first page.")
            else:
                print("No content pages found in the parsed document.")
                
        else:
            print(f"Error: Expected output file not found at {output_json_path}")
            
    except Exception as e:
        print(f"Error during PDF parsing: {e}")
        print("This could be due to issues with the PDF file or model incompatibilities.")
        print("Ensure all models for PDFParser components are correctly downloaded and configured.")
        
        # Fallback: try basic PDF info extraction
        print("Attempting basic PDF analysis...")
        try:
            from pdfminer.high_level import extract_pages
            page_count = 0
            for _ in extract_pages(sample_pdf_path):
                page_count += 1
            print(f"Basic check: pdfminer.six detected {page_count} pages in the document.")
        except Exception as pe:
            print(f"Error during basic pdfminer.six check: {pe}")

if __name__ == "__main__":
    main()