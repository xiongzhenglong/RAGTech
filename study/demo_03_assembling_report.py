# study/demo_03_assembling_report.py

import json
import os

def main():
    """
    Demonstrates how the parsed PDF JSON output is assembled,
    focusing on the conceptual role of `JsonReportProcessor`.
    """
    print("Starting demonstration of report assembly (conceptual)...")

    # --- The Role of JsonReportProcessor ---
    # When `src.pdf_parsing.PDFParser.parse_and_export()` is called, it internally
    # uses a backend (like `DoclingParseV2DocumentBackend`) to process the PDF.
    # The raw output from this backend (often a `ConversionResult` object) is then
    # passed to `src.pdf_parsing.JsonReportProcessor`.
    #
    # `JsonReportProcessor` is responsible for taking this raw, detailed output
    # and organizing it into a structured and more accessible JSON format.
    # This involves:
    #   - `assemble_metainfo()`: Extracts document-level properties (filename, SHA, page count).
    #   - `assemble_content()`: Organizes text blocks, paragraphs, headers, footers,
    #     and other content elements for each page, resolving references.
    #   - `assemble_tables()`: Processes raw table data (cells, structure) and often generates
    #     multiple representations (e.g., Markdown, HTML, a structured JSON list of cells).
    #   - `assemble_pictures()`: Processes raw image/figure data, extracting bounding boxes
    #     and other relevant details.
    #
    # The JSON file we are loading below is the direct output of this assembly process,
    # orchestrated by `PDFParser` using `JsonReportProcessor`.

    # --- 1. Define Input JSON Path ---
    # This is the path to the JSON file generated by demo_01_pdf_parsing.py,
    # which itself is a result of PDFParser's internal use of JsonReportProcessor.
    input_json_filename = "194000c9109c6fa628f1fed33b44ae4c2b8365f4.json"
    input_json_path = os.path.join("study", "parsed_output", input_json_filename)
    print(f"Attempting to load parsed data from: {input_json_path}")

    # --- 2. Load JSON Data ---
    if not os.path.exists(input_json_path):
        print(f"Error: Parsed JSON file not found at {input_json_path}")
        print("Please ensure you have run 'demo_01_pdf_parsing.py' successfully.")
        return

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            parsed_data = json.load(f)
        print("Successfully loaded parsed JSON data.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode the JSON file at {input_json_path}.")
        return
    except Exception as e:
        print(f"An error occurred while loading the JSON file: {e}")
        return

    # --- 3. Re-examine Sections (Focus on Assembly by JsonReportProcessor) ---

    # --- 3.1. Metainfo ---
    # The `metainfo` section is generated by `JsonReportProcessor.assemble_metainfo()`.
    # This method would take raw document properties from the parsing backend's output
    # (e.g., document hash, original filename, page count) and structure them.
    print("\n--- Document Metainfo (Assembled by JsonReportProcessor.assemble_metainfo) ---")
    if 'metainfo' in parsed_data and parsed_data['metainfo']:
        for key, value in parsed_data['metainfo'].items():
            print(f"  {key}: {value}")
    else:
        print("  No 'metainfo' section found.")
    print("--------------------------------------------------------------------------------")

    # --- 3.2. Content of the First Page ---
    # The `content` section is assembled by `JsonReportProcessor.assemble_content()`.
    # This method processes the raw output for each page, which might include lists of
    # text lines, layout elements, and their coordinates. `assemble_content()` organizes
    # these into a hierarchical structure of pages, blocks (paragraphs, headers, etc.),
    # and text spans. It also handles resolving internal document references if present.
    print("\n--- Content of First Page (Assembled by JsonReportProcessor.assemble_content) ---")
    if 'content' in parsed_data and parsed_data['content']:
        first_page_data = parsed_data['content'][0]
        page_number = first_page_data.get('page', 'N/A')
        print(f"  Page Number: {page_number}")

        page_text_segments = []
        if 'content' in first_page_data and first_page_data['content']:
            for block in first_page_data['content']:
                if 'text' in block and block['text']:
                    page_text_segments.append(str(block['text']))
                elif 'content' in block and block['content']:
                    for sub_element in block['content']:
                        if 'text' in sub_element and sub_element['text']:
                             page_text_segments.append(str(sub_element['text']))
        if page_text_segments:
            full_page_text = " ".join(page_text_segments)
            print(f"  Combined Text (Snippet):\n\"{full_page_text[:500]}...\"")
            print("  (This structured text is a result of assemble_content() processing raw text and layout data.)")
        else:
            print("  No textual content found for the first page in the expected structure.")
    else:
        print("  No 'content' section found.")
    print("-----------------------------------------------------------------------------------")

    # --- 3.3. Details of the First Table ---
    # The `tables` section is created by `JsonReportProcessor.assemble_tables()`.
    # The parsing backend (e.g., Docling) would provide raw table detection results,
    # possibly as lists of cells with their text and coordinates. `assemble_tables()`
    # transforms this into a more usable format, including:
    #   - Table ID, page number, bounding box.
    #   - Number of rows and columns.
    #   - Multiple representations like Markdown, HTML, and a structured JSON list of cells/rows.
    print("\n--- Details of the First Table (Assembled by JsonReportProcessor.assemble_tables) ---")
    if 'tables' in parsed_data and parsed_data['tables']:
        first_table_data = parsed_data['tables'][0]
        print(f"  Table ID: {first_table_data.get('table_id', 'N/A')}")
        print(f"  Page: {first_table_data.get('page', 'N/A')}")
        print(f"  Dimensions: {first_table_data.get('rows', 'N/A')} rows x {first_table_data.get('columns', 'N/A')} columns")
        markdown_repr = first_table_data.get('markdown', 'N/A')
        print(f"  Markdown Representation (Snippet):\n{markdown_repr[:300]}...")
        print("  (This structured table object, including various formats, is generated by assemble_tables().)")
    else:
        print("  No tables found in the document.")
    print("---------------------------------------------------------------------------------------")

    # --- 3.4. Details of the First Picture ---
    # The `pictures` section is assembled by `JsonReportProcessor.assemble_pictures()`.
    # The backend provides raw data about detected images/figures (e.g., coordinates, possibly raw image bytes or references).
    # `assemble_pictures()` standardizes this into a list of picture objects with IDs, page numbers, and bounding boxes.
    print("\n--- Details of the First Picture (Assembled by JsonReportProcessor.assemble_pictures) ---")
    if 'pictures' in parsed_data and parsed_data['pictures']:
        first_picture_data = parsed_data['pictures'][0]
        print(f"  Picture ID: {first_picture_data.get('picture_id', 'N/A')}") # Or 'id'
        print(f"  Page: {first_picture_data.get('page', 'N/A')}")
        print(f"  Bounding Box (bbox): {first_picture_data.get('bbox', 'N/A')}")
        print("  (This structured picture object is generated by assemble_pictures().)")
    else:
        print("  No pictures found in the document.")
    print("-----------------------------------------------------------------------------------------")

    # --- 4. Conceptual Difference from Raw Backend Output ---
    # If one were to use a backend like `DoclingParseV2DocumentBackend().convert_all()` directly,
    # the output would typically be a more complex, less immediately usable object (e.g., a `ConversionResult` object).
    # This raw output would contain highly detailed information, including:
    #   - Extensive coordinate data for every detected element (lines, words, characters).
    #   - Raw text without much semantic grouping (e.g., not explicitly identified as paragraphs).
    #   - Table data as raw cell content and structure, without pre-generated Markdown/HTML.
    #   - Potentially multiple candidate interpretations for layout elements.
    #
    # `JsonReportProcessor` acts as a crucial post-processing step. It consumes this
    # raw `ConversionResult` and applies logic to:
    #   - Simplify and structure the data.
    #   - Aggregate related information (e.g., group lines into paragraphs).
    #   - Generate convenient representations (like Markdown for tables).
    #   - Select the most probable interpretation of the document structure.
    # The JSON file we are observing is this refined, application-ready output.
    print("\n--- Conceptual Note on Raw Backend Output vs. Assembled JSON ---")
    print("The JSON data explored here is a *processed and structured representation* of the PDF's content.")
    print("`JsonReportProcessor` (used internally by `PDFParser`) transforms raw, highly detailed")
    print("output from the core parsing engine (e.g., Docling models) into this more organized format.")
    print("Direct output from the backend would be more granular and less directly usable for many applications.")
    print("----------------------------------------------------------------")

    print("\nDemonstration of report assembly concepts complete.")

if __name__ == "__main__":
    main()
