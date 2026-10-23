"""
PDF Ingestion Module for Gas Turbine Maintenance Assistant.
Loads GER-3620-P.pdf, cleans and chunks text while preserving page and section metadata,
and builds or updates the persistent ChromaDB vector collection.
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    PDF_PATH,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# Known primary section headings from GER-3620 table of contents
KNOWN_SECTIONS = [
    "Introduction",
    "Maintenance Planning",
    "Digital Solutions for Asset Management",
    "Gas Turbine Configuration Maintenance Features",
    "Borescope Inspections",
    "Major Factors Influencing Maintenance and Equipment Life",
    "Starts and Hours Criteria",
    "Service Factors",
    "Fuel",
    "Firing Temperatures",
    "Steam/Water Injection",
    "Cyclic Effects and Fast Starts",
    "Hot Gas Path Parts",
    "Rotor Parts",
    "Combustion Parts",
    "Casing Parts",
    "Exhaust Diffuser Parts",
    "Off-Frequency Operation",
    "Compressor Condition and Performance",
    "Lube Oil Cleanliness",
    "Moisture Intake",
    "Maintenance Inspections",
    "Standby Inspections",
    "Running Inspections",
    "Rapid Cool-Down",
    "Combustion Inspection",
    "Hot Gas Path Inspection",
    "Major Inspection",
    "Parts Planning",
    "Inspection Intervals",
    "Borescope Inspection Interval",
    "ABC Inspections For Annular/Silo Fleet",
    "Combustion Inspection Interval",
    "Hot Gas Path Inspection Interval",
    "Rotor Inspection Interval",
    "GE Vernova Annular/Silo Fleet Welded Rotor Inspection",
    "Personnel Planning",
    "Conclusion",
    "References",
    "Appendix",
]


def clean_page_text(raw_text: str, physical_page: int) -> tuple[str, Optional[int]]:
    """
    Cleans repetitive PDF headers and footers while preserving technical content,
    tables, numbers, equations, and section headings.
    Returns (cleaned_text, doc_page_number).
    """
    lines = raw_text.splitlines()
    cleaned_lines = []
    doc_page = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        # Filter standard repeating header
        if re.search(r"Doc Name\s*:\s*GER-3620", stripped, re.IGNORECASE):
            continue
        # Filter standard classification footer
        if re.search(r"Non-Public.*Class II", stripped, re.IGNORECASE):
            continue
        # Filter document title banner when standalone
        if stripped == "Heavy-Duty Gas Turbine Operating and Maintenance Considerations":
            continue

        # Check if line is purely the document page number (e.g. '1', '21', 'ii')
        if re.match(r"^\d{1,2}$", stripped) and doc_page is None:
            doc_page = int(stripped)
            continue
        if re.match(r"^[ivxIVX]{1,4}$", stripped) and doc_page is None:
            # Roman numeral for front matter
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, doc_page


def detect_section(page_text: str, current_section: str) -> str:
    """
    Detects if any major section heading appears on the page and updates
    the current section context.
    """
    lower_text = page_text.lower()
    for section in KNOWN_SECTIONS:
        # Check if section title is on a line by itself or in heading form
        pattern = rf"(?m)^\s*{re.escape(section)}\s*$"
        if re.search(pattern, page_text, re.IGNORECASE):
            return section
    return current_section


def load_and_process_pdf(pdf_path: Path) -> List[Document]:
    """
    Loads GER-3620-P.pdf using PyMuPDF, extracts text with metadata,
    and splits into chunks.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found at: {pdf_path}. "
            "Please ensure GER-3620-P.pdf is in the 'data/' directory."
        )

    print(f"📖 Opening PDF: {pdf_path}")
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)
    print(f"📄 Total pages in document: {total_pages}")

    raw_docs: List[Document] = []
    current_section = "Introduction"

    for page_idx in range(total_pages):
        page = doc[page_idx]
        physical_page = page_idx + 1
        raw_text = page.get_text()

        if not raw_text.strip():
            print(f"⚠️ Page {physical_page} is empty or unreadable, skipping.")
            continue

        cleaned_text, doc_page = clean_page_text(raw_text, physical_page)
        current_section = detect_section(cleaned_text, current_section)

        # Fallback doc page estimation if not explicitly parsed from page header/footer
        # In GER-3620-P.pdf: Page 5 is Doc Page 1
        if doc_page is None and physical_page >= 5:
            doc_page = physical_page - 4

        raw_docs.append(
            Document(
                page_content=cleaned_text,
                metadata={
                    "source": pdf_path.name,
                    "page": physical_page,
                    "doc_page": doc_page if doc_page is not None else physical_page,
                    "section": current_section,
                },
            )
        )

    print(f"✅ Extracted text from {len(raw_docs)} non-empty pages.")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )

    chunked_docs: List[Document] = []
    chunk_counter = 0

    for doc_item in raw_docs:
        chunks = text_splitter.split_text(doc_item.page_content)
        for c_idx, chunk_text in enumerate(chunks):
            chunk_counter += 1
            chunk_id = f"ger3620_p{doc_item.metadata['page']}_c{c_idx}"
            chunked_docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **doc_item.metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": c_idx,
                    },
                )
            )

    print(f"🧩 Split into {len(chunked_docs)} semantic chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunked_docs


def ingest_document(pdf_path: Path = PDF_PATH, force: bool = False) -> int:
    """
    Ingests document into persistent ChromaDB.
    Skips if collection already populated, unless force=True.
    Returns the count of chunks indexed.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"⚙️ Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Check if existing collection has documents
    if not force and CHROMA_DIR.exists():
        try:
            existing_store = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embedding_function,
                collection_name=CHROMA_COLLECTION_NAME,
            )
            count = len(existing_store.get()["ids"])
            if count > 0:
                print(f"ℹ️ ChromaDB already contains {count} chunks. Use --force to re-ingest.")
                return count
        except Exception as e:
            print(f"Note: Could not check existing collection ({e}), proceeding with fresh ingestion.")

    # Reset or create directory if forced
    if force and CHROMA_DIR.exists():
        print("🔄 Force re-ingestion requested. Clearing existing vector store...")
        import shutil
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Process PDF
    chunks = load_and_process_pdf(pdf_path)
    if not chunks:
        raise ValueError("No chunks were produced from PDF processing.")

    print(f"💾 Storing {len(chunks)} chunks into ChromaDB at '{CHROMA_DIR}'...")
    ids = [c.metadata["chunk_id"] for c in chunks]

    # Batch insertion to optimize memory and progress
    batch_size = 64
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_function,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        vectorstore.add_documents(documents=batch_chunks, ids=batch_ids)
        print(f"  Indexed chunks {i + 1} to {min(i + batch_size, len(chunks))} of {len(chunks)}")

    print(f"🎉 Ingestion successfully completed! Total indexed chunks: {len(chunks)}")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest GER-3620-P.pdf into ChromaDB")
    parser.add_argument(
        "--force", action="store_true", help="Force re-ingestion even if database exists"
    )
    parser.add_argument(
        "--pdf-path", type=str, default=str(PDF_PATH), help="Path to PDF file"
    )
    args = parser.parse_args()

    target_pdf = Path(args.pdf_path)
    try:
        count = ingest_document(target_pdf, force=args.force)
        print(f"Summary: {count} chunks ready in ChromaDB.")
    except Exception as err:
        print(f"❌ Ingestion failed: {err}", file=sys.stderr)
        sys.exit(1)
