AI Document Agent – Hybrid LLM + Reasoning System
Overview

This project is a prototype AI system designed to answer questions from complex technical documents.

It combines large language models (LLMs) with structured reasoning to improve accuracy, especially when working with tables and numerical data.

The goal is to build a more reliable and practical AI assistant for real-world engineering and enterprise use cases.

Key Features
1. Document Understanding
Reads PDF documents
Extracts both text and tables
Filters out sensitive or restricted content
2. Retrieval-Augmented Generation (RAG)
Converts document content into embeddings
Retrieves only the most relevant sections for a query
Ensures answers are grounded in source material
3. Hybrid Reasoning System
Uses structured (rule-based) reasoning for:
table queries
numeric comparisons (e.g., max capacity, lowest latency)
Uses an LLM for:
natural language understanding
general question answering
4. Hallucination Detection
Checks whether generated answers are supported by retrieved content
Triggers prompt refinement when needed
5. Self-Improving Prompt Loop
Uses a simple reinforcement learning (RL) mechanism
Tracks which prompting strategies perform better
Adjusts prompts over time
Example

Question:
Which SSD has the largest capacity?

System Behavior:

Extracts table data from document
Applies structured reasoning (max comparison)

Answer:
D5-P5336 has the largest capacity (61TB)

Architecture
User Query
    ↓
Retriever (RAG)
    ↓
Agent
 ├── Structured Reasoning (tables, numeric logic)
 ├── Prompt Selection (RL-based)
 ├── LLM Generation
 └── Hallucination Detection
    ↓
Final Answer
Why This Approach?

LLMs alone can:

struggle with tables and numbers
generate incorrect or hallucinated answers

This system improves reliability by:

combining deterministic logic with LLMs
grounding responses in retrieved documents
validating outputs before returning them
Tech Stack
Python
Hugging Face Transformers
Sentence Transformers (embeddings)
PDF parsing (pdfplumber)
Future Improvements
Stronger LLM integration
Improved hallucination detection
Scalable deployment (API / service architecture)
Better evaluation metrics
Author

Hugo Lin
M.S. Computer Science (NLP / AI), UC Santa Cruz
