import random  # Import the random module for random choices

# List of different prompt templates the agent can use
prompts = [
    "Answer using only the provided context.",
    "Use step-by-step reasoning from the context.",
    "Extract information carefully from tables if present.",
    "Verify each claim against the retrieved documents."
]

# Initialize a dictionary to track the "quality" or score (Q-value) of each prompt
# Initially, all prompts have a score of 0
Q = {p: 0 for p in prompts}

# -----------------------------
# Function to choose which prompt to use for a query
# -----------------------------
def choose_prompt():

    # With 20% probability, pick a random prompt (exploration)
    if random.random() < 0.2:
        return random.choice(prompts)

    # Otherwise, pick the prompt with the highest score (exploitation)
    return max(Q, key=Q.get)

# -----------------------------
# Function to update the reward for a prompt after seeing how well it performed
# -----------------------------
def update_reward(prompt, reward, lr=0.1):
    """
    prompt: the prompt used
    reward: numerical feedback indicating how good the prompt was (e.g., 1 for correct answer, 0 for hallucination)
    lr: learning rate to control how fast we update the prompt score
    """

    # Update the Q-value using a simple reinforcement learning formula
    # Q_new = Q_old + lr * (reward - Q_old)
    Q[prompt] = Q[prompt] + lr * (reward - Q[prompt])