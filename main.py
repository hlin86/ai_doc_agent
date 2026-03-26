from ingest import extract_document, embed_documents, model
from retriever import VectorStore
from agent import ask_agent

from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large"
)

def real_llm(prompt, query, docs):

    context = "\n".join(docs)

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

    output = generator(full_prompt, max_new_tokens=150)

    return output[0]["generated_text"]

# -----------------------------
# 1. Load and Process Document
# -----------------------------

file = "technical_doc.pdf"

documents = extract_document(file, mode="filter")

store = VectorStore()

embeddings, texts = embed_documents(documents)

store.add(texts, embeddings)


# -----------------------------
# 2. Query Embedding Function
# -----------------------------

def embed_query(query):
    return model.encode([query])[0]


# -----------------------------
# 3. Interactive Loop
# -----------------------------

while True:

    query = input("Ask a question: ")

    answer = ask_agent(query, store, embed_query, real_llm)

    print("\nAnswer:\n", answer)