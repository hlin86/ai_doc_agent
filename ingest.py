import pdfplumber
from sentence_transformers import SentenceTransformer

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 1. Sensitive Content Settings
# -----------------------------

SENSITIVE_KEYWORDS = [
    "internal only",
    "do not share",
    "confidential",
    "failed test",
    "debug",
    "not for customer"
]

def detect_sensitive(text):
    text_lower = text.lower()
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in text_lower:
            return True
    return False


# -----------------------------
# 2. Convert Tables to JSON-like Structure
# -----------------------------

def table_to_json(table):
    """
    Converts a table (list of rows) into structured dict format
    """
    if not table or len(table) < 2:
        return None

    headers = table[0]
    rows = table[1:]

    structured = []

    for row in rows:
        item = {}
        for h, cell in zip(headers, row):
            item[h] = cell
        structured.append(item)

    return structured


# -----------------------------
# 3. Main Extraction Function
# -----------------------------

def extract_document(file, mode="filter"):
    """
    mode:
        - "filter": remove sensitive content
        - "tag": keep but label sensitive content
    """

    documents = []

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:

            # -------- TEXT --------
            text = page.extract_text()

            if text:
                is_sensitive = detect_sensitive(text)

                if mode == "filter":
                    if not is_sensitive:
                        documents.append({
                            "type": "text",
                            "content": text,
                            "access": "public"
                        })

                elif mode == "tag":
                    documents.append({
                        "type": "text",
                        "content": text,
                        "access": "internal" if is_sensitive else "public"
                    })

            # -------- TABLES --------
            tables = page.extract_tables()

            for table in tables:
                structured = table_to_json(table)

                if structured:
                    table_text = str(structured)

                    is_sensitive = detect_sensitive(table_text)

                    if mode == "filter":
                        if not is_sensitive:
                            documents.append({
                                "type": "table",
                                "content": structured,
                                "access": "public"
                            })

                    elif mode == "tag":
                        documents.append({
                            "type": "table",
                            "content": structured,
                            "access": "internal" if is_sensitive else "public"
                        })

    return documents


# -----------------------------
# 4. Embedding Function
# -----------------------------

def embed_documents(documents):

    texts = []

    for doc in documents:

        if doc["type"] == "text":
            texts.append(doc["content"])

        elif doc["type"] == "table":
            texts.append(str(doc["content"]))  # convert structured table to string

    embeddings = model.encode(texts)

    return embeddings, texts