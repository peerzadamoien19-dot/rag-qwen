from indexer import embed_texts


def search(question, index, chunks, top_k=3):
    """Find the chunks most relevant to the question."""
    question_vector = embed_texts([question])

    distances, positions = index.search(question_vector, top_k)

    results = []
    for i in positions[0]:
        results.append(chunks[i])

    return results
  

if __name__ == "__main__":
    from ingest import read_pdf, make_chunks
    from indexer import build_index

    text = read_pdf("test.pdf")
    chunks = make_chunks(text)

    print("Building index, please wait...")
    index = build_index(chunks)

    question = "how do I copy a file"
    results = search(question, index, chunks)

    print("\nQuestion:", question)
    print("\n--- Top matches ---\n")

    for r in results:
        print(r)
        print("\n---\n")  