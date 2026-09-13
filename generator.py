from huggingface_hub import InferenceClient

MODEL = "Qwen/Qwen3.8-27B"
PROVIDER = "novita"


def build_prompt(question, chunks):
    """Combine the retrieved chunks and the question into one prompt."""
    context = "\n\n".join(chunks)

    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    return prompt


def ask_qwen(question, chunks, token):
    """Send the prompt to Qwen and return its answer."""
    client = InferenceClient(api_key=token, provider=PROVIDER)
    prompt = build_prompt(question, chunks)

    response = client.chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    import tomllib

    from ingest import read_pdf, make_chunks
    from indexer import build_index
    from retriever import search

    with open(".streamlit/secrets.toml", "rb") as f:
        token = tomllib.load(f)["HF_TOKEN"]

    text = read_pdf("test.pdf")
    chunks = make_chunks(text)

    print("Building index, please wait...")
    index = build_index(chunks)

    question = "how do I copy a file"
    found = search(question, index, chunks)

    print("Asking Qwen...")
    answer = ask_qwen(question, found, token)

    print("\nQuestion:", question)
    print("\nAnswer:\n")
    print(answer)