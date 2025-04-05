import streamlit as st
import faiss
import pickle
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
from datetime import datetime

st.set_page_config(
    page_title="Legal Document Generator",
    page_icon="⚖️",
    layout="wide"
)


# Load the SBERT model
@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("sbert_model")


# Load FAISS index
@st.cache_resource
def load_faiss_index():
    return faiss.read_index("faiss_index.faiss")


# Load metadata
@st.cache_resource
def load_metadata():
    with open("ipc_data.pkl", "rb") as f:
        return pickle.load(f)


# Load LLM model for document generation
@st.cache_resource
def load_text2text_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )

    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
    return generator


# App title and intro
st.title("Legal Document Generator ⚖️")
st.markdown("""
This application helps legal professionals find relevant IPC sections for a case 
and generate appropriate legal documents based on case descriptions.
""")

# Sidebar for app navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select Mode",
    ["IPC Section Search", "Legal Document Generator"]
)

# Initialize session state for selected sections
if 'selected_sections' not in st.session_state:
    st.session_state.selected_sections = []

if 'search_results' not in st.session_state:
    st.session_state.search_results = []

# Load models and data
with st.spinner('Loading models and data...'):
    try:
        sbert_model = load_sbert_model()
        index = load_faiss_index()
        data = load_metadata()

        # Extract data from the loaded pickle
        sections = data["sections"]
        offenses = data["offenses"]
        punishments = data["punishments"]
        descriptions = data["descriptions"]

        # Only load text2text model when needed
        generator = None
    except Exception as e:
        st.error(f"Error loading models or data: {e}")
        st.stop()


# Function to find the most relevant IPC section
def find_ipc_section(case_description, k=3):
    query_embedding = sbert_model.encode([case_description], convert_to_numpy=True)
    D, I = index.search(query_embedding, k)  # Get top-k matches

    results = []
    for i in range(k):
        sec_num = sections[I[0][i]]
        sec_desc = descriptions[I[0][i]]
        offense = offenses[I[0][i]]
        punishment = punishments[I[0][i]]
        results.append((sec_num, offense, punishment, sec_desc, D[0][i]))

    return results


# Function to toggle section selection
def toggle_section(index):
    result = st.session_state.search_results[index]

    # Check if this section is already in selected_sections
    for i, section in enumerate(st.session_state.selected_sections):
        if section[0] == result[0]:  # Compare section numbers
            # If found, remove it
            st.session_state.selected_sections.pop(i)
            return

    # If not found, add it
    st.session_state.selected_sections.append(result)


# Function to generate legal document using Flan-T5
def generate_legal_document(case_description, accused_name, selected_sections):
    # Load Flan-T5 model on demand if not already loaded
    global generator
    if generator is None:
        with st.spinner('Loading language model for document generation...'):
            generator = load_text2text_model()

    # Create sections list for prompt
    sections_list = [f"Section {sec}" for sec, _, _, _, _ in selected_sections]
    sections_str = ", ".join(sections_list)

    # Create more detailed context from selected sections
    legal_context = ""
    for sec, offense, punishment, desc, _ in selected_sections:
        if not isinstance(offense, str) or offense.lower() == 'nan':
            offense = "Not specified"
        if not isinstance(punishment, str) or punishment.lower() == 'nan':
            punishment = "Not specified"

        legal_context += f"Section {sec}:\n"
        legal_context += f"- Offense: {offense}\n"
        legal_context += f"- Punishment: {punishment}\n"
        legal_context += f"- Description: {desc}\n\n"

    # Create prompt template for Flan-T5
    prompt = f"""
    Based on the case description: "{case_description}"

    Accused person: {accused_name}

    And the following applicable IPC sections:
    {legal_context}

    Generate a formal First Information Report (FIR) that:
    1. Provides details of the alleged offense
    2. Explains how each IPC section applies to the case facts
    3. Recommends appropriate legal action

    Write in professional legal language suitable for an official FIR document.
    """

    # Generate text using the Flan-T5 model
    try:
        response = generator(prompt, max_length=1024, do_sample=True, temperature=0.7)[0]['generated_text']

        # Create a more structured FIR with the generated content
        fir = f"""
FIRST INFORMATION REPORT
========================

FIR No: [FIR Number]/2025
Police Station: [Station Name]
District: [District]
State: [State]
Date: {datetime.now().strftime('%d-%m-%Y')}

COMPLAINANT DETAILS:
-------------------
Name: [Complainant Name]
Address: [Address]
Contact: [Contact Details]

ACCUSED PERSON DETAILS:
---------------------
Name: {accused_name}
Address: [Address]
Other Identifiable Information: [Details if available]

{response}

APPLICABLE IPC SECTIONS:
----------------------
{", ".join(sections_list)}

Signed,
[Investigating Officer]
[Rank and Badge Number]
[Police Station]
"""
        return fir

    except Exception as e:
        st.error(f"Error generating document with Flan-T5: {e}")
        return generate_legal_document_template_based(case_description, accused_name, selected_sections)


# Template-based document generation function
def generate_legal_document_template_based(case_description, accused_name, selected_sections):
    """
    Generate a legal document using a template-based approach with minimal LLM requirements
    """
    if not selected_sections:
        return "No IPC sections selected. Please select at least one section."

    # Extract the best matching IPC section
    primary_section = selected_sections[0]
    sec_num, offense, punishment, desc, _ = primary_section

    # Handle NaN values
    if not isinstance(offense, str) or str(offense).lower() == 'nan':
        offense = "Criminal Act"
    if not isinstance(punishment, str) or str(punishment).lower() == 'nan':
        punishment = "As per applicable law"

    # Create additional sections text
    additional_sections = ""
    for i in range(1, len(selected_sections)):
        sec, off, pun, _, _ = selected_sections[i]
        if isinstance(off, str) and str(off).lower() != 'nan':
            additional_sections += f"- Section {sec}: {off}\n"

    # Template for FIR
    template = f"""
FIRST INFORMATION REPORT
========================

FIR No: [FIR Number]/2025
Police Station: [Station Name]
District: [District]
State: [State]
Date: {datetime.now().strftime('%d-%m-%Y')}

COMPLAINANT DETAILS:
-------------------
Name: [Complainant Name]
Address: [Address]
Contact: [Contact Details]

ACCUSED PERSON DETAILS:
---------------------
Name: {accused_name}
Address: [Address]
Other Identifiable Information: [Details if available]

INCIDENT DETAILS:
---------------
Date and Time of Incident: [Date and Time]
Place of Incident: [Location]

DESCRIPTION OF OFFENSE:
---------------------
As per the complaint received, it is alleged that {case_description}.

SECTIONS OF LAW APPLIED:
----------------------
1. Primary Section: {sec_num} - {offense}
   Justification: The accused's actions constitute an offense under Section {sec_num} of the Indian Penal Code as they involve {offense.lower() if offense.lower() != "nan" else "a criminal act as defined by this section"}.

{additional_sections}

PUNISHMENT APPLICABLE:
--------------------
{punishment}

ACTION TAKEN:
-----------
Based on the complaint and preliminary investigation, an FIR has been registered. The investigation has commenced to ascertain the facts of the case. The accused shall be apprehended for questioning and further legal procedures shall be followed according to the Code of Criminal Procedure.

Signed,
[Investigating Officer]
[Rank and Badge Number]
[Police Station]
"""
    return template


# Mode 1: IPC Section Search
if app_mode == "IPC Section Search":
    st.header("IPC Section Finder 🔍")
    st.write("Enter a case description, and we'll find the most relevant IPC sections.")

    case_description = st.text_area("Enter Case Description (e.g., 'murder', 'theft')")
    k = st.slider("Number of Matches", 1, 10, 3)

    search_button = st.button("Find IPC Section")

    if search_button:
        if case_description.strip():
            with st.spinner('Searching for relevant IPC sections...'):
                matches = find_ipc_section(case_description, k)
                st.session_state.search_results = matches

    # Display search results if available
    if st.session_state.search_results:
        st.subheader("Search Results:")

        # Create a table to display results
        col_headings = st.columns([1, 2, 5, 2])
        col_headings[0].markdown("**Select**")
        col_headings[1].markdown("**Section**")
        col_headings[2].markdown("**Offense**")
        col_headings[3].markdown("**Relevance**")

        st.markdown("---")

        # Check which sections are already selected
        selected_section_numbers = [sec[0] for sec in st.session_state.selected_sections]

        for i, (sec, offense, punishment, desc, dist) in enumerate(st.session_state.search_results):
            cols = st.columns([1, 2, 5, 2])

            # Check if this section is already in selected sections
            is_selected = sec in selected_section_numbers

            # Display checkbox in first column
            if cols[0].checkbox("", value=is_selected, key=f"select_{i}_{sec}"):
                if not is_selected:
                    st.session_state.selected_sections.append(st.session_state.search_results[i])
            else:
                if is_selected:
                    # Find and remove from selected sections
                    for j, selected_section in enumerate(st.session_state.selected_sections):
                        if selected_section[0] == sec:
                            st.session_state.selected_sections.pop(j)
                            break

            # Display section number in second column
            cols[1].write(f"{sec}")

            # Display offense in third column
            cols[2].write(f"{offense if isinstance(offense, str) else 'Not specified'}")

            # Display relevance score in fourth column
            relevance = 1 - (dist / 10)
            cols[3].write(f"{relevance:.2f}")

            # Add an expander for more details
            with st.expander(f"Details for Section {sec}"):
                st.write(f"**Offense:** {offense if isinstance(offense, str) else 'Not specified'}")
                st.write(f"**Punishment:** {punishment if isinstance(punishment, str) else 'Not specified'}")
                st.write(f"**Description:** {desc}")

        # Summary of selected sections
        st.subheader("Selected Sections")
        if st.session_state.selected_sections:
            st.write(f"{len(st.session_state.selected_sections)} sections selected:")
            for sec, offense, _, _, _ in st.session_state.selected_sections:
                st.write(f"- Section {sec}: {offense if isinstance(offense, str) else 'Not specified'}")
        else:
            st.info("No sections selected yet. Select sections from the results above.")

# Mode 2: Legal Document Generator
elif app_mode == "Legal Document Generator":
    st.header("Legal Document Generator 📝")
    st.write("Generate a legal document based on detailed case description and selected IPC sections.")

    col1, col2 = st.columns([3, 1])

    with col1:
        detailed_case_description = st.text_area(
            "Detailed Case Description",
            height=150,
            help="Enter a detailed case description (e.g., 'Dhruv murdered a girl by stabbing her with a knife')"
        )

    with col2:
        accused_name = st.text_input("Accused Name")
        document_type = st.selectbox(
            "Document Type",
            ["First Information Report (FIR)", "Charge Sheet", "Legal Notice"]
        )
        use_template = st.checkbox(
            "Use Template-based Generation",
            value=True,
            help="Use template instead of LLM. Faster but less customized."
        )

    # Section for viewing and managing selected IPC sections
    st.subheader("Selected IPC Sections")

    if not st.session_state.selected_sections:
        st.info(
            "No IPC sections selected yet. Use the Quick Search below or go to 'IPC Section Search' to select relevant sections.")
    else:
        # Display selected sections with option to remove
        col_headings = st.columns([1, 3, 5, 1])
        col_headings[0].markdown("**#**")
        col_headings[1].markdown("**Section**")
        col_headings[2].markdown("**Offense**")
        col_headings[3].markdown("**Action**")

        st.markdown("---")

        for i, (sec, offense, punishment, desc, _) in enumerate(st.session_state.selected_sections):
            cols = st.columns([1, 3, 5, 1])

            cols[0].write(f"{i + 1}")
            cols[1].write(f"{sec}")
            cols[2].write(f"{offense if isinstance(offense, str) else 'Not specified'}")

            if cols[3].button("Remove", key=f"remove_{i}"):
                st.session_state.selected_sections.pop(i)
                st.rerun()

        # Option to clear all sections
        if st.button("Clear All Sections"):
            st.session_state.selected_sections = []
            st.rerun()

    # Quick Search feature
    with st.expander("Quick Search for IPC Sections"):
        quick_search = st.text_input("Enter keywords (e.g., 'murder', 'theft')")
        col1, col2 = st.columns([1, 3])
        quick_k = col1.slider("Number of results", 1, 5, 3)
        search_clicked = col2.button("Quick Search")

        if search_clicked and quick_search.strip():
            with st.spinner('Searching...'):
                quick_matches = find_ipc_section(quick_search, quick_k)

            # Display results in a table
            col_headings = st.columns([1, 2, 5, 2])
            col_headings[0].markdown("**Select**")
            col_headings[1].markdown("**Section**")
            col_headings[2].markdown("**Offense**")
            col_headings[3].markdown("**Relevance**")

            st.markdown("---")

            # Check which sections are already selected
            selected_section_numbers = [sec[0] for sec in st.session_state.selected_sections]

            for i, (sec, offense, punishment, desc, dist) in enumerate(quick_matches):
                cols = st.columns([1, 2, 5, 2])

                # Check if this section is already in selected sections
                is_selected = sec in selected_section_numbers

                # Display checkbox in first column
                if cols[0].checkbox("", value=is_selected, key=f"quick_select_{i}_{sec}"):
                    if not is_selected:
                        # Check if this section is already in selected sections
                        already_selected = False
                        for selected_sec in st.session_state.selected_sections:
                            if selected_sec[0] == sec:
                                already_selected = True
                                break

                        if not already_selected:
                            st.session_state.selected_sections.append(quick_matches[i])
                else:
                    if is_selected:
                        # Find and remove from selected sections
                        for j, selected_section in enumerate(st.session_state.selected_sections):
                            if selected_section[0] == sec:
                                st.session_state.selected_sections.pop(j)
                                break

                # Display section number in second column
                cols[1].write(f"{sec}")

                # Display offense in third column
                cols[2].write(f"{offense if isinstance(offense, str) else 'Not specified'}")

                # Display relevance score in fourth column
                relevance = 1 - (dist / 10)
                cols[3].write(f"{relevance:.2f}")

    # Generate document button
    if st.button("Generate Document"):
        if detailed_case_description.strip() and accused_name.strip():
            if not st.session_state.selected_sections:
                st.warning("No IPC sections selected. Please select at least one section.")
            else:
                # Generate document
                with st.spinner('Generating legal document...'):
                    if use_template:
                        document = generate_legal_document_template_based(
                            detailed_case_description,
                            accused_name,
                            st.session_state.selected_sections
                        )
                    else:
                        try:
                            document = generate_legal_document(
                                detailed_case_description,
                                accused_name,
                                st.session_state.selected_sections
                            )
                        except Exception as e:
                            st.error(f"Error using Flan-T5: {e}")
                            st.info("Falling back to template-based approach...")
                            document = generate_legal_document_template_based(
                                detailed_case_description,
                                accused_name,
                                st.session_state.selected_sections
                            )

                # Display the generated document
                st.subheader("Generated Document")
                st.text_area("Document Content", document, height=400)

                # Download button for the document
                st.download_button(
                    label="Download Document",
                    data=document,
                    file_name=f"legal_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Please enter both detailed case description and accused name.")

# Footer
st.markdown("---")
st.markdown("**Legal Document Generator** - Powered by NLP and Machine Learning")