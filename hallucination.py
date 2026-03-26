# Function to detect if an answer contains hallucinations
# Hallucinations are statements not supported by the retrieved context
def hallucination_detect(answer, context):

    # Combine all documents or retrieved context into a single lowercase string
    context_text = " ".join(context).lower()

    # Split the answer into sentences by periods
    sentences = answer.split(".")

    # Counter for sentences that are not supported by the context
    unsupported = 0

    # Loop through each sentence in the answer
    for s in sentences:

        # Skip very short sentences (less than 5 characters) to avoid counting punctuation/noise
        if len(s.strip()) < 5:
            continue

        # Check if the sentence (lowercased) is not found in the context
        if s.lower() not in context_text:
            unsupported += 1  # Increment unsupported counter

    # If more than one sentence is unsupported, consider the answer a hallucination
    if unsupported > 1:
        return True

    # Otherwise, the answer is supported by the context
    return False