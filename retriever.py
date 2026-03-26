# Import NumPy library for numerical operations, especially vectors
import numpy as np

# -----------------------------
# VectorStore Class
# -----------------------------
# Simple in-memory vector store for embeddings and associated documents
class VectorStore:

    # Constructor to initialize empty lists for embeddings and documents
    def __init__(self):
        self.embeddings = []  # List to store embeddings (vectors)
        self.documents = []   # List to store corresponding documents/texts

    # Method to add documents and their embeddings to the store
    def add(self, docs, embeddings):

        # Loop through each document and its corresponding embedding
        for d, e in zip(docs, embeddings):
            self.documents.append(d)    # Add document to documents list
            self.embeddings.append(e)   # Add embedding to embeddings list

    # Method to search for the top-k most similar documents given a query embedding
    def search(self, query_embedding, k=3):

        sims = []  # List to store similarity scores

        # Loop through each embedding in the store
        for emb in self.embeddings:
            # Compute cosine similarity between query_embedding and current embedding
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            sims.append(sim)  # Store similarity score

        # Get the indices of the top-k highest similarity scores
        idx = np.argsort(sims)[-k:]

        # Return the documents corresponding to the top-k indices
        return [self.documents[i] for i in idx]