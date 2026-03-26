# Import library to read PDF files
import pdfplumber

# Import SentenceTransformer for embedding text into vectors
from sentence_transformers import SentenceTransformer

# -----------------------------
# 0. Load Embedding Model
# -----------------------------

# Load a small, efficient model for embedding text (sentence-level embeddings)
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 1. Sensitive Content Settings
# -----------------------------

# List of keywords indicating content should be restricted
SENSITIVE_KEYWORDS = [
    "internal only",
    "do not share",
    "confidential",
    "failed test",
    "debug",
    "not for customer"
]

# Function to detect whether a text contains sensitive content
def detect_sensitive(text):
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    # Check each keyword in the list
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in text_lower:
            return True  # If any keyword matches, mark as sensitive
    return False  # No keywords found, text is not sensitive

# -----------------------------
# 2. Convert Tables to JSON-like Structure
# -----------------------------

# Function to turn tables into a structured Python dictionary format
def table_to_json(table):
    """
    Converts a table (list of rows) into structured dict format
    """
    if not table or len(table) < 2:
        return None  # Skip if table is empty or has no rows

    # First row assumed to be headers
    headers = table[0]
    # Remaining rows are data
    rows = table[1:]

    structured = []  # List to store each row as a dictionary

    # Iterate over each row
    for row in rows:
        item = {}  # dictionary for this row
        # Map each cell to its header
        for h, cell in zip(headers, row):
            item[h] = cell
        structured.append(item)  # add row dictionary to list

    return structured  # Return structured table

# -----------------------------
# 3. Main Extraction Function
# -----------------------------

# Function to read a PDF and return processed documents
def extract_document(file, mode="filter"):
    """
    Extracts text and tables from a PDF.

    mode:
        - "filter": remove sensitive content completely
        - "tag": keep sensitive content but label it as internal
    """

    documents = []  # List to store extracted documents

    # Open PDF file
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:  # Loop through each page

            # -------- TEXT --------
            text = page.extract_text()  # Extract text from page

            if text:
                is_sensitive = detect_sensitive(text)  # Check if text is sensitive

                if mode == "filter":
                    if not is_sensitive:  # Skip sensitive text
                        documents.append({
                            "type": "text",
                            "content": text,
                            "access": "public"
                        })

                elif mode == "tag":  # Keep sensitive text but label it
                    documents.append({
                        "type": "text",
                        "content": text,
                        "access": "internal" if is_sensitive else "public"
                    })

            # -------- TABLES --------
            tables = page.extract_tables()  # Extract tables from page

            for table in tables:
                structured = table_to_json(table)  # Convert to structured dict

                if structured:
                    table_text = str(structured)  # Convert to string for keyword check

                    is_sensitive = detect_sensitive(table_text)  # Check sensitivity

                    if mode == "filter":
                        if not is_sensitive:  # Skip sensitive tables
                            documents.append({
                                "type": "table",
                                "content": structured,
                                "access": "public"
                            })

                    elif mode == "tag":  # Keep table but label access
                        documents.append({
                            "type": "table",
                            "content": structured,
                            "access": "internal" if is_sensitive else "public"
                        })

    return documents  # Return all extracted and processed documents

# -----------------------------
# 4. Embedding Function
# -----------------------------

# Function to convert all document text/tables into embeddings
def embed_documents(documents):

    texts = []  # Store text representations of each document

    # Convert each document to a string representation
    for doc in documents:

        if doc["type"] == "text":
            texts.append(doc["content"])  # Use raw text

        elif doc["type"] == "table":
            texts.append(str(doc["content"]))  # Convert structured table to string

    embeddings = model.encode(texts)  # Generate embeddings for all texts

    return embeddings, texts  # Return embeddings and the original texts