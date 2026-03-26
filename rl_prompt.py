import random

prompts = [
    "Answer using only the provided context.",
    "Use step-by-step reasoning from the context.",
    "Extract information carefully from tables if present.",
    "Verify each claim against the retrieved documents."
]

Q = {p: 0 for p in prompts}


def choose_prompt():

    if random.random() < 0.2:
        return random.choice(prompts)

    return max(Q, key=Q.get)


def update_reward(prompt, reward, lr=0.1):

    Q[prompt] = Q[prompt] + lr * (reward - Q[prompt])