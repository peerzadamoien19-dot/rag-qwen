import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts):
    """Turn a list of texts into a list of number vectors."""
    vectors = model.encode(texts)
    return vectors


def build_index(chunks):
    """Create a FAISS index from text chunks."""
    vectors = embed_texts(chunks)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index
  
  
if __name__ == "__main__":
    from ingest import read_pdf, make_chunks

    text = read_pdf("test.pdf")
    chunks = make_chunks(text)

    print("Chunks:", len(chunks))
    print("Building index... this may take a minute")

    index = build_index(chunks)

    print("Vectors stored:", index.ntotal)