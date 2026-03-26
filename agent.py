# Import functions for prompt selection/updating and hallucination checking
from rl_prompt import choose_prompt, update_reward
from hallucination import hallucination_detect

# -----------------------------
# Structured Reasoning Layer
# -----------------------------
def structured_reasoning(query, docs):
    """
    A deterministic (rule-based) reasoning layer for numeric or table queries.
    Tries to answer the question without calling an LLM if possible.
    """

    import ast  # Import abstract syntax tree module to safely parse strings into Python objects

    try:
        data = []  # Initialize list to hold parsed table data
        for d in docs:  # Iterate through each document chunk
            if d.strip().startswith("["):  # Check if document looks like a table (JSON-like list)
                data.extend(ast.literal_eval(d))  # Safely convert string to Python list/dict
    except:
        return None  # If parsing fails, return None

    query = query.lower()  # Lowercase the query for simple string matching

    # Rule-based reasoning for "capacity" queries
    if "capacity" in query:
        # Find the row with max capacity
        best = max(data, key=lambda x: int(x["Capacity"].replace("TB", "")))
        return f"{best['Model']} has the largest capacity ({best['Capacity']})."

    # Rule-based reasoning for "latency" queries
    if "latency" in query:
        # Find the row with minimum latency
        best = min(data, key=lambda x: int(x["Latency"].replace("us", "")))
        return f"{best['Model']} has the lowest latency ({best['Latency']})."

    return None  # Return None if no rule-based answer is found

# -----------------------------
# Main Agent Function
# -----------------------------
def ask_agent(query, store, embed_query, llm):
    """
    The main agent function that processes a query end-to-end:
    1. Retrieves relevant documents
    2. Tries structured reasoning
    3. Calls an LLM if needed
    4. Checks for hallucination
    5. Updates RL prompt loop
    """

    # Step 1: Retrieve relevant documents using vector store
    query_emb = embed_query(query)  # Convert query to embedding
    docs = store.search(query_emb)  # Retrieve top-matching documents

    # Step 2: Try structured reasoning first (faster, deterministic)
    structured_answer = structured_reasoning(query, docs)

    if structured_answer:
        print("[Using structured reasoning]")  # Debug message for demo
        return structured_answer  # Return immediately if a structured answer exists

    # Step 3: Choose which prompt to use (RL-based prompt selection)
    prompt = choose_prompt()

    # Step 4: Call the LLM with the chosen prompt, query, and retrieved docs
    answer = llm(prompt, query, docs)

    # Step 5: Check if LLM answer is hallucinated (not supported by docs)
    is_hallucinated = hallucination_detect(answer, docs)

    if is_hallucinated:
        print("[Hallucination detected → refining prompt]")  # Debug message
        refined_prompt = prompt + " Cite the supporting text."  # Refine prompt to reduce hallucination
        answer = llm(refined_prompt, query, docs)  # Re-call LLM with refined prompt
        reward = -1  # Negative reward for RL update
    else:
        reward = 1  # Positive reward if no hallucination

    # Step 6: Update RL prompt scores based on performance
    update_reward(prompt, reward)

    # Debug info (useful during demos)
    print("\n--- DEBUG ---")
    print("Prompt:", prompt)
    print("Hallucination:", is_hallucinated)
    print("-------------\n")

    return answer  # Return the final answer to the user