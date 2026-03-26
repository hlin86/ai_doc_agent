import numpy as np

class VectorStore:

    def __init__(self):
        self.embeddings = []
        self.documents = []

    def add(self, docs, embeddings):

        for d, e in zip(docs, embeddings):
            self.documents.append(d)
            self.embeddings.append(e)

    def search(self, query_embedding, k=3):

        sims = []

        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            sims.append(sim)

        idx = np.argsort(sims)[-k:]

        return [self.documents[i] for i in idx]