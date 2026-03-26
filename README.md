AI Document Agent – Hybrid LLM + Reasoning System

Overview

This project is a prototype AI system designed to answer questions from complex technical documents.

It combines large language models (LLMs) with structured (rule-based) reasoning to improve accuracy, especially when working with tables and numerical data.

The goal is to build a more reliable and practical AI assistant for real-world engineering and enterprise use cases.

Key Features

1. Document Understanding

Reads PDF documents

Extracts both text and tables

Supports filtering of sensitive or restricted content during ingestion

2. Retrieval-Augmented Generation (RAG)

Converts document content into embeddings

Retrieves the most relevant sections for a query

Grounds answers in retrieved document context

3. Hybrid Reasoning System

Uses structured (rule-based) reasoning for:

table queries

numeric comparisons (e.g., max capacity, lowest latency)

Uses an LLM for:

natural language understanding

general question answering

This hybrid approach improves correctness for structured data while maintaining flexibility for open-ended queries.

4. Basic Hallucination Detection

Applies simple heuristic checks to determine whether an answer is supported by retrieved content

If a potential issue is detected, the system refines the prompt and retries

Note: This is a lightweight prototype and not a fully robust hallucination detection system.

5. Prompt Optimization (Prototype)

Uses a simple reward-based mechanism to select between prompting strategies

Adjusts prompt usage within a single run based on observed outcomes

Note: This is an early-stage prototype and does not persist learning across sessions.

Example

Question:

Which SSD has the largest capacity?

System Behavior:

Extracts table data from the document

Applies structured reasoning (max comparison)

Answer:

D5-P5336 has the largest capacity (61TB)

Architecture

High-Level Flow

User asks a question

System retrieves relevant document sections

Agent generates an answer using reasoning and validation

Detailed Flow

User Query

    ↓

Retriever (RAG)

    ↓

Agent

 ├── Structured Reasoning (tables, numeric logic)

 ├── Prompt Selection (prototype RL)

 ├── LLM Generation

 └── Hallucination Check (heuristic)

    ↓

Final Answer

Why This Approach?

LLMs alone can:

struggle with tables and numeric comparisons

generate incorrect or unsupported answers

This system improves reliability by:

combining deterministic logic with LLM reasoning

grounding responses in retrieved documents

adding lightweight validation before returning answers

Tech Stack

Python

Hugging Face Transformers

Sentence Transformers (embeddings)

PDF parsing (pdfplumber)

Limitations

Hallucination detection is heuristic-based and not fully robust

Prompt optimization does not persist across runs

Performance depends on the quality of retrieved document chunks

Small/medium LLMs may produce less detailed responses

Future Improvements

Stronger LLM integration

More robust hallucination detection and evaluation

Persistent feedback loops for prompt optimization

Scalable deployment (API / service architecture)

Author

Hugo Lin

M.S. Computer Science (NLP / AI), UC Santa Cruz
