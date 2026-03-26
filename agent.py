from rl_prompt import choose_prompt, update_reward
from hallucination import hallucination_detect


# -----------------------------
# Structured Reasoning Layer
# -----------------------------
def structured_reasoning(query, docs):

    import ast

    try:
        data = []
        for d in docs:
            if d.strip().startswith("["):
                data.extend(ast.literal_eval(d))
    except:
        return None

    query = query.lower()

    if "capacity" in query:
        best = max(data, key=lambda x: int(x["Capacity"].replace("TB", "")))
        return f"{best['Model']} has the largest capacity ({best['Capacity']})."

    if "latency" in query:
        best = min(data, key=lambda x: int(x["Latency"].replace("us", "")))
        return f"{best['Model']} has the lowest latency ({best['Latency']})."

    return None


# -----------------------------
# Main Agent Function
# -----------------------------
def ask_agent(query, store, embed_query, llm):

    # Step 1: Retrieve documents
    query_emb = embed_query(query)
    docs = store.search(query_emb)

    # Step 2: Try structured reasoning FIRST
    structured_answer = structured_reasoning(query, docs)

    if structured_answer:
        print("[Using structured reasoning]")
        return structured_answer

    # Step 3: Choose prompt (RL)
    prompt = choose_prompt()

    # Step 4: Call LLM
    answer = llm(prompt, query, docs)

    # Step 5: Hallucination detection
    is_hallucinated = hallucination_detect(answer, docs)

    if is_hallucinated:
        print("[Hallucination detected → refining prompt]")
        refined_prompt = prompt + " Cite the supporting text."
        answer = llm(refined_prompt, query, docs)
        reward = -1
    else:
        reward = 1

    # Step 6: Update RL
    update_reward(prompt, reward)

    # Debug (VERY useful for demo)
    print("\n--- DEBUG ---")
    print("Prompt:", prompt)
    print("Hallucination:", is_hallucinated)
    print("-------------\n")

    return answer