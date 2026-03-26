def hallucination_detect(answer, context):

    context_text = " ".join(context).lower()

    sentences = answer.split(".")

    unsupported = 0

    for s in sentences:

        if len(s.strip()) < 5:
            continue

        if s.lower() not in context_text:
            unsupported += 1

    if unsupported > 1:
        return True

    return False