import os
import re
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("resources/data")

# Engineering is already uploaded to Pinecone
DEPARTMENTS = ["finance", "general", "hr", "marketing"]

MAX_TOKENS = 500
CHUNK_OVERLAP = 50

INDEX_NAME = "ds-rpc-01"
TARGET_DIMENSION = 1024


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env")


# ============================================================
# TOKENIZER
# ============================================================

encoding = tiktoken.get_encoding("cl100k_base")


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"}
)

print("Embedding model loaded.")


# ============================================================
# CONNECT TO PINECONE
# ============================================================

pc = Pinecone(api_key=PINECONE_API_KEY)


# ============================================================
# CREATE INDEX IF IT DOES NOT EXIST
# ============================================================

existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:

    print(f"Creating Pinecone index: {INDEX_NAME}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=TARGET_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    print("Index created.")

else:

    print(f"Pinecone index already exists: {INDEX_NAME}")


# ============================================================
# GET INDEX
# ============================================================

index = pc.Index(INDEX_NAME)


# ============================================================
# CREATE CHUNKS FROM CONTENT
# ============================================================


def create_chunks(content, heading_stack, department, document_title):
    """
    Split content into token-based chunks.

    Each chunk contains:

    Department
    Document
    Parent
    Heading
    Content
    """

    if not content.strip():
        return []

    tokens = encoding.encode(content)

    chunks = []

    start = 0

    # --------------------------------------------------------
    # Current heading
    # --------------------------------------------------------

    current_heading = (
        heading_stack[-1]
        if heading_stack
        else {"number": None, "level": 0, "title": ""}
    )

    # --------------------------------------------------------
    # Parent heading
    # --------------------------------------------------------

    parent_heading = heading_stack[-2] if len(heading_stack) >= 2 else None

    # --------------------------------------------------------
    # Split content into overlapping chunks
    # --------------------------------------------------------

    while start < len(tokens):

        end = min(start + MAX_TOKENS, len(tokens))

        chunk_tokens = tokens[start:end]

        chunk_content = encoding.decode(chunk_tokens).strip()

        if chunk_content:

            # ------------------------------------------------
            # Build text that will actually be embedded
            # ------------------------------------------------

            chunk_text = f"""Department: {department}
Document: {document_title}
Parent: {parent_heading["title"] if parent_heading else ""}
Heading: {current_heading["title"]}

{chunk_content}"""

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = {
                "department": department,
                "document_title": document_title,
                "parent_title": (parent_heading["title"] if parent_heading else ""),
                "heading_title": current_heading["title"],
                "text": chunk_text,
            }

            chunks.append({"text": chunk_text, "metadata": metadata})

        # ----------------------------------------------------
        # Stop after final chunk
        # ----------------------------------------------------

        if end == len(tokens):
            break

        # ----------------------------------------------------
        # Move forward while keeping overlap
        # ----------------------------------------------------

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# PROCESS ONE MARKDOWN DOCUMENT
# ============================================================


def process_document(file_path, department):
    """
    Read one Markdown document and convert it into chunks.

    Supports:

    Numbered headings:

        ## 2. System Architecture
        ### 2.2 High-Level Architecture
        #### 2.2.1 Components

    Normal Markdown headings:

        # Quarterly Financial Report
        ## Executive Summary
        ## Q1
        ### Quarterly Financial Overview
    """

    print(f"\nProcessing: {department}/{file_path.name}")

    with open(file_path, "r", encoding="utf-8") as file:

        lines = file.readlines()

    document_title = ""

    heading_stack = []

    content_buffer = []

    all_chunks = []

    # ========================================================
    # REGEX
    # ========================================================

    # Matches numbered headings such as:
    #
    # ## 2. System Architecture
    # ### 2.2 High-Level Architecture
    # #### 2.2.1 Components

    numbered_heading_pattern = re.compile(r"^(#{1,6})\s+(\d+(?:\.\d+)*)\.?\s+(.*)$")

    # Matches normal Markdown headings such as:
    #
    # # Financial Report
    # ## Executive Summary
    # ### Quarterly Overview

    markdown_heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

    # ========================================================
    # FLUSH CURRENT CONTENT
    # ========================================================

    def flush_content():

        nonlocal content_buffer

        if not content_buffer:
            return

        content = "\n".join(content_buffer).strip()

        if content:

            chunks = create_chunks(
                content=content,
                heading_stack=heading_stack,
                department=department,
                document_title=document_title,
            )

            all_chunks.extend(chunks)

        content_buffer = []

    # ========================================================
    # PROCESS LINES
    # ========================================================

    for raw_line in lines:

        line = raw_line.strip()

        # ----------------------------------------------------
        # Ignore empty lines
        # ----------------------------------------------------

        if not line:
            continue

        # ----------------------------------------------------
        # Try numbered heading first
        # ----------------------------------------------------

        numbered_match = numbered_heading_pattern.match(line)

        if numbered_match:

            hashes = numbered_match.group(1)

            number = numbered_match.group(2)

            title = numbered_match.group(3).strip()

            # Level based on number hierarchy
            #
            # 2       -> 1
            # 2.2     -> 2
            # 2.2.1   -> 3

            level = number.count(".") + 1

            # ------------------------------------------------
            # Document title
            # ------------------------------------------------

            if level == 1 and not document_title:

                flush_content()

                document_title = title

                continue

            # ------------------------------------------------
            # Normal heading
            # ------------------------------------------------

            flush_content()

            heading = {"number": number, "level": level, "title": title}

            # Remove headings at same or deeper level
            while heading_stack and heading_stack[-1]["level"] >= level:
                heading_stack.pop()

            heading_stack.append(heading)

            continue

        # ----------------------------------------------------
        # Try normal Markdown heading
        # ----------------------------------------------------

        markdown_match = markdown_heading_pattern.match(line)

        if markdown_match:

            hashes = markdown_match.group(1)

            title = markdown_match.group(2).strip()

            level = len(hashes)

            # ------------------------------------------------
            # Level 1 = document title
            # ------------------------------------------------

            if level == 1 and not document_title:

                flush_content()

                document_title = title

                continue

            # ------------------------------------------------
            # Normal heading
            # ------------------------------------------------

            flush_content()

            heading = {"number": None, "level": level, "title": title}

            while heading_stack and heading_stack[-1]["level"] >= level:
                heading_stack.pop()

            heading_stack.append(heading)

            continue

        # ----------------------------------------------------
        # Normal content
        # ----------------------------------------------------

        content_buffer.append(line)

    # ========================================================
    # PROCESS REMAINING CONTENT
    # ========================================================

    flush_content()

    print(f"Created {len(all_chunks)} chunks " f"from {file_path.name}")

    return all_chunks


# ============================================================
# MAIN INGESTION
# ============================================================

all_chunks = []


for department in DEPARTMENTS:

    department_path = DATA_PATH / department

    if not department_path.exists():

        print(f"\nWARNING: Directory does not exist: " f"{department_path}")

        continue

    markdown_files = sorted(department_path.glob("*.md"))

    if not markdown_files:

        print(f"\nWARNING: No Markdown files found " f"for {department}")

        continue

    print(f"\n{'=' * 60}")

    print(f"DEPARTMENT: {department}")

    print(f"FILES: {len(markdown_files)}")

    print(f"{'=' * 60}")

    # --------------------------------------------------------
    # Process EVERY file in this department
    # --------------------------------------------------------

    for file_path in markdown_files:

        department_chunks = process_document(file_path=file_path, department=department)

        all_chunks.extend(department_chunks)


# ============================================================
# CHECK IF THERE IS ANYTHING TO UPLOAD
# ============================================================

if not all_chunks:

    print("\nNo chunks were created.")

    raise SystemExit


print(f"\nTotal chunks created: {len(all_chunks)}")


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating embeddings...")

texts = [chunk["text"] for chunk in all_chunks]

vectors = embeddings.embed_documents(texts)

print(f"Generated {len(vectors)} embeddings.")

print(f"Vector dimension: {len(vectors[0])}")


# ============================================================
# CREATE VECTOR RECORDS
# ============================================================

vectors_to_upsert = []

# IMPORTANT:
#
# Counter is maintained PER DEPARTMENT.
#
# This means:
#
# finance file 1:
#   finance-0
#   finance-1
#   finance-2
#
# finance file 2:
#   finance-3
#   finance-4
#   finance-5
#
# So IDs never collide between files.

department_counters = {}


for i, chunk in enumerate(all_chunks):

    department = chunk["metadata"]["department"]

    # --------------------------------------------------------
    # Initialize counter for department
    # --------------------------------------------------------

    if department not in department_counters:

        department_counters[department] = 0

    # --------------------------------------------------------
    # Generate ID
    # --------------------------------------------------------

    chunk_number = department_counters[department]

    vector_id = f"{department}-{chunk_number}"

    department_counters[department] += 1

    # --------------------------------------------------------
    # Create Pinecone record
    # --------------------------------------------------------

    vectors_to_upsert.append(
        {"id": vector_id, "values": vectors[i], "metadata": chunk["metadata"]}
    )


# ============================================================
# SHOW COUNTS
# ============================================================

print("\nVectors to upload:")

for department, count in department_counters.items():

    print(f"  {department}: {count}")


# ============================================================
# UPSERT INTO PINECONE
# ============================================================

print("\nUploading vectors to Pinecone...")

index.upsert(vectors=vectors_to_upsert)

print("\nUpload completed successfully.")


# ============================================================
# FINAL INDEX STATS
# ============================================================

stats = index.describe_index_stats()

print("\nPinecone index statistics:")

print(stats)
