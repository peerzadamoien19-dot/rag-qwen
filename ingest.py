from pypdf import PdfReader


def read_pdf(file):
    """Pull all the text out of a PDF."""
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text = text + page_text + "\n"

    return text


def make_chunks(text, chunk_size=500, overlap=100):
    """Cut long text into overlapping pieces."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks
  
   
if __name__ == "__main__":
    text = read_pdf("test.pdf")
    chunks = make_chunks(text)

    print("Characters extracted:", len(text))
    print("Chunks created:", len(chunks))
    print("\nFirst chunk:\n")
    print(chunks[0])