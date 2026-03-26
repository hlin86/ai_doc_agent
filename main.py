# -----------------------------
# Imports
# -----------------------------
# Functions to extract text/tables and convert them into embeddings
from ingest import extract_document, embed_documents, model
# Vector store to hold document embeddings for retrieval
from retriever import VectorStore
# Agent that handles structured reasoning, LLM calls, hallucination detection, and RL
from agent import ask_agent

# Hugging Face pipeline for using a pre-trained LLM
from transformers import pipeline

# -----------------------------
# 1. Initialize LLM
# -----------------------------
generator = pipeline(
    "text2text-generation",      # Task: convert input text into output text
    model="google/flan-t5-large" # Model: Google Flan-T5 large (text-to-text)
)

# -----------------------------
# 2. Define Real LLM Function
# -----------------------------
def real_llm(prompt, query, docs):
    """
    Calls the actual LLM instead of fake/hard-coded responses.
    Combines context (retrieved docs) and query into a single prompt.
    """

    context = "\n".join(docs)  # Merge all retrieved document chunks into one string

    # Construct a structured prompt with instructions + context + question
    full_prompt = f"""
You are a technical assistant.

Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't know."
Answer in a complete sentence.
Be precise and include relevant details like model names, capacity, and interface.

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate answer using the Hugging Face pipeline
    output = generator(full_prompt, max_new_tokens=150)  # Limit output length

    # Return the generated text from the first result
    return output[0]["generated_text"]

# -----------------------------
# 3. Load and Process Document
# -----------------------------
file = "technical_doc.pdf"  # Path to the PDF file to read

# Extract text and tables, removing sensitive content
documents = extract_document(file, mode="filter")

# Create an empty vector store
store = VectorStore()

# Convert documents into embeddings and retrieve corresponding texts
embeddings, texts = embed_documents(documents)

# Add embeddings + texts into the vector store for retrieval
store.add(texts, embeddings)

# -----------------------------
# 4. Query Embedding Function
# -----------------------------
def embed_query(query):
    """
    Convert user query into an embedding for vector similarity search.
    """
    return model.encode([query])[0]  # Encode single query and return embedding

# -----------------------------
# 5. Interactive Loop
# -----------------------------
while True:
    # Ask user for a query
    query = input("Ask a question: ")

    # Get answer from the hybrid agent (structured reasoning + LLM + hallucination checks)
    answer = ask_agent(query, store, embed_query, real_llm)

    # Display the answer
    print("\nAnswer:\n", answer)