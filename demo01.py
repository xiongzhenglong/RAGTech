import pandas as pd

def load_expertise_from_excel(file_path="专家库.xlsx"):
    """
    Loads expertise terms from the first column of the first sheet of an Excel file.

    Args:
        file_path (str, optional): The path to the Excel file.
                                     Defaults to "专家库.xlsx".

    Returns:
        list: A list of expertise terms, or an empty list if an error occurs.
    """
    try:
        df = pd.read_excel(file_path, header=None, sheet_name=0)
        # Assuming expertise terms are in the first column (index 0)
        expertise_terms = df[0].dropna().unique().tolist()
        return expertise_terms
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Returning empty list.")
        return []
    except Exception as e:
        print(f"An error occurred while parsing the Excel file: {e}")
        return []

if __name__ == '__main__':
    print("Attempting to load expertise from Excel...")
    # This will print a warning if "专家库.xlsx" is not found, which is expected.
    expertise_list = load_expertise_from_excel()

    if not expertise_list:
        print("No expertise terms loaded from Excel. Using a sample list for demonstration.")
        # Provide a sample expertise list if the Excel load fails or returns empty
        # This helps in demonstrating process_resume_text even without the actual Excel file.
        expertise_list = ["Python", "Java", "Machine Learning", "Data Analysis", "Cloud Computing", "Project Management", "Natural Language Processing", "Text Analytics"]
        print(f"Using sample expertise list: {expertise_list}")
    else:
        print(f"Successfully loaded expertise terms: {expertise_list}")


    # Define a clear, multi-line sample resume_text string
    sample_resume = """
    Dr. Eleanor Vance
    Senior Data Scientist | Machine Learning Expert

    Summary:
    Highly accomplished Data Scientist with 12 years of experience in developing and implementing
    state-of-the-art machine learning models. Proven ability to translate complex data into
    actionable insights. Expertise in Python, R, and SQL. Strong background in Natural Language Processing (NLP)
    and cloud computing platforms like AWS and Azure. Seeking to leverage data-driven strategies
    to solve challenging problems.

    Experience:
    Lead Data Scientist, Tech Solutions Inc. (2018 - Present)
      - Developed predictive models for customer churn, resulting in a 15% reduction.
      - Implemented a recommendation engine using collaborative filtering and content-based approaches.
      - Specialized in Text Analytics and Natural Language Processing for sentiment analysis.
    Data Scientist, Data Corp (2012 - 2018)
      - Worked on various data analysis projects for clients in the retail and finance sectors.
      - Utilized Java for legacy system integration.

    Skills:
    - Programming: Python (Expert), R (Proficient), SQL (Advanced), Java (Intermediate)
    - Machine Learning: Regression, Classification, Clustering, Deep Learning, NLP
    - Tools: Scikit-learn, TensorFlow, Keras, Pandas, NumPy, Spark
    - Cloud Platforms: AWS, Azure, GCP
    - Other: Data Visualization, Statistical Analysis, Project Management
    """

    print(f"\nProcessing sample resume against the expertise list...")
    # Call process_resume_text with the sample resume and (loaded or sample) expertise list
    matched_terms = process_resume_text(sample_resume, expertise_list)

    # Print the returned matched_expertise with a descriptive message
    if matched_terms:
        print("\n--- Matched Expertise Terms in Resume ---")
        for term in matched_terms:
            print(f"- {term}")
        print("--------------------------------------")
    else:
        print("\nNo expertise terms from the list were found in the sample resume.")

# --- Integration Notes for resume_analyzer.py ---
# The functions in this file (load_expertise_from_excel and process_resume_text)
# are designed to be integrated into a larger resume analysis pipeline, specifically
# a system like 'resume_analyzer.py' that might currently use an LLM for parsing.
#
# 1. Location for Integration:
#    - The logic from `process_resume_text` (after loading expertise via `load_expertise_from_excel`)
#      could be called within functions like `analyze_resume_file` or `analyze_uploaded_content`
#      in `resume_analyzer.py`.
#    - Alternatively, `load_expertise_from_excel` could be called once at the start of
#      `resume_analyzer.py` (or when it's first needed) to load the expertise list,
#      and then `process_resume_text` could be used as a helper function.
#
# 2. Input to the New Logic:
#    - The resume text, once extracted by `resume_analyzer.py` (e.g., from a PDF or DOCX using
#      `extract_text_from_pdf` or `extract_text_from_docx`), would be the `resume_text`
#      argument for `process_resume_text`.
#    - The `expertise_list` would be obtained by calling `load_expertise_from_excel`.
#
# 3. Using the Output:
#    - The list of `matched_expertise` returned by `process_resume_text` can be used to
#      populate the "专长" (expertise) field in the JSON output of `resume_analyzer.py`.
#    - This might involve modifying the part of `resume_analyzer.py` that currently formats
#      the LLM output, to instead use or include these matched terms.
#
# 4. Excel File Management:
#    - The "专家库.xlsx" file must be accessible to `resume_analyzer.py`. This might mean
#      placing it in a predefined directory relative to `resume_analyzer.py`, or providing
#      its path via a configuration setting.
#    - Error handling in `load_expertise_from_excel` should be considered in the context
#      of the main application (e.g., log errors, notify admin if file is missing).
#
# 5. Replacing/Augmenting LLM Call for Expertise:
#    - Design Choice: This new logic could either:
#      a) Completely replace the LLM's role in identifying expertise terms. This would make
#         the "专长" section fully dependent on the contents of "专家库.xlsx".
#      b) Augment the LLM. For example:
#         - Use the `matched_expertise` from this script as the primary source for "专长".
#           If the list is short or empty, the LLM could be queried to find additional terms.
#         - Run both this logic and the LLM in parallel and then combine the results
#           (e.g., take the union of both sets of terms, or prioritize one over the other).
#         - Use this logic for the "专长" field and continue to use the LLM for other
#           fields in the resume JSON (like "工作经历", "教育背景", etc.).
#    - The best approach depends on the desired accuracy, consistency, and the specific
#      strengths/weaknesses of the LLM for expertise extraction versus a predefined list.
#
# Example call within resume_analyzer.py (conceptual):
#
#   # In resume_analyzer.py, after extracting resume_text:
#   # global_expert_list = demo01.load_expertise_from_excel("path/to/专家库.xlsx") # Load once
#   # ...
#   # extracted_resume_text = extract_text_from_pdf(file_path)
#   # matched_skills = demo01.process_resume_text(extracted_resume_text, global_expert_list)
#   # resume_json_data["专长"] = matched_skills
#   # ... rest of LLM processing for other fields or as fallback ...
#

def process_resume_text(resume_text: str, expertise_list: list):
    """
    Processes resume text against a list of expertise terms,
    identifies matches, and returns them.

    Args:
        resume_text (str): The text of the resume.
        expertise_list (list): A list of expertise terms.

    Returns:
        list: A list of expertise terms found in the resume_text.
    """
    normalized_resume_text = resume_text.lower()
    normalized_expertise_list = [term.lower() for term in expertise_list]

    matched_expertise = []

    print("\n--- Processing Resume ---") # Keep print for now for clarity
    print("Original Resume Text (first 100 chars):", resume_text[:100] + "...")
    print("Normalized Expertise List:", normalized_expertise_list)

    for term in normalized_expertise_list:
        if term in normalized_resume_text:
            # Add the original term (not the normalized one) for better readability
            original_term = expertise_list[normalized_expertise_list.index(term)]
            if original_term not in matched_expertise: # Avoid duplicates if original list had case variations
                 matched_expertise.append(original_term)

    print("--- End Processing ---")
    return matched_expertise
