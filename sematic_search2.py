import boto3
import json
import numpy as np
from numpy.linalg import norm
import os
import PyPDF2
import docx

# -----------------------------
# 1. Bedrock Client
# -----------------------------
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)
MODEL_ID = "amazon.titan-embed-text-v2:0"
TOP_K = 3
MIN_SCORE = 0.3
DOCUMENTS_FILE = "documents.json"
EMBEDDINGS_FILE = "embeddings.json"

# -----------------------------
# 2. Generate Embedding
# -----------------------------
def get_embedding(text):
    try:
        body = {"inputText": text}
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
    except Exception as e:
        raise RuntimeError(f"Bedrock error: {e}")

# -----------------------------
# 3. Cosine Similarity
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# -----------------------------
# 4. Load / Save Documents
# -----------------------------
def load_documents():
    if not os.path.exists(DOCUMENTS_FILE):
        return []
    with open(DOCUMENTS_FILE, "r") as f:
        return json.load(f)

def save_documents(docs):
    with open(DOCUMENTS_FILE, "w") as f:
        json.dump(docs, f, indent=2)

# -----------------------------
# 5. Load / Save Embeddings
# -----------------------------
def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        return []
    with open(EMBEDDINGS_FILE, "r") as f:
        return json.load(f)

def save_embeddings(embeddings):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings, f)

# -----------------------------
# 6. Create / Refresh Embeddings
# -----------------------------
def create_embeddings():
    docs = load_documents()
    if not docs:
        print("❌ No documents found in documents.json!")
        return

    existing = {item["text"]: item["embedding"] for item in load_embeddings()}
    embeddings = []

    for doc in docs:
        if doc in existing:
            print(f"⏭️  Skipping (already embedded): {doc[:60]}...")
            embeddings.append({"text": doc, "embedding": existing[doc]})
        else:
            print(f"🔄 Embedding: {doc[:60]}...")
            emb = get_embedding(doc)
            embeddings.append({"text": doc, "embedding": emb})

    save_embeddings(embeddings)
    print(f"✅ Embeddings saved! Total: {len(embeddings)} documents.")

# -----------------------------
# 7. Search Function
# -----------------------------
def search(query):
    if not os.path.exists(EMBEDDINGS_FILE):
        print("❌ No embeddings found. Run option 1 first!")
        return []

    query_embedding = get_embedding(query)
    stored = load_embeddings()

    similarities = [
        (item["text"], float(cosine_similarity(
            np.array(query_embedding),
            np.array(item["embedding"])
        )))
        for item in stored
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [(t, s) for t, s in similarities[:TOP_K] if s >= MIN_SCORE]

# -----------------------------
# 8. Add Document
# -----------------------------
def add_document(text):
    text = text.strip()
    if not text:
        print("❌ Document text cannot be empty.")
        return

    docs = load_documents()
    if text in docs:
        print("⚠️  Document already exists!")
        return

    docs.append(text)
    save_documents(docs)

    # Immediately embed the new doc
    embeddings = load_embeddings()
    print(f"🔄 Embedding new document...")
    emb = get_embedding(text)
    embeddings.append({"text": text, "embedding": emb})
    save_embeddings(embeddings)
    print(f"✅ Document added and embedded!")

# -----------------------------
# 9. Delete Document
# -----------------------------
def delete_document():
    docs = load_documents()
    if not docs:
        print("❌ No documents to delete.")
        return

    print("\n📋 Current Documents:")
    for i, doc in enumerate(docs):
        print(f"  {i+1}. {doc[:80]}...")

    try:
        choice = int(input("\nEnter document number to delete (0 to cancel): "))
        if choice == 0:
            return
        if choice < 1 or choice > len(docs):
            print("❌ Invalid choice.")
            return

        removed = docs.pop(choice - 1)
        save_documents(docs)

        # Remove from embeddings too
        embeddings = [e for e in load_embeddings() if e["text"] != removed]
        save_embeddings(embeddings)
        print(f"✅ Deleted: {removed[:60]}...")
    except ValueError:
        print("❌ Please enter a valid number.")

# -----------------------------
# 10. List All Documents
# -----------------------------
def list_documents():
    docs = load_documents()
    if not docs:
        print("❌ No documents found.")
        return
    print(f"\n📋 All Documents ({len(docs)} total):")
    for i, doc in enumerate(docs):
        print(f"  {i+1}. {doc[:80]}")

# -----------------------------
# 11. Extract Text from PDF
# -----------------------------
def extract_from_pdf(filepath):
    texts = []
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    # Split into chunks of ~500 chars
                    chunks = [text[j:j+500] for j in range(0, len(text), 500)]
                    for chunk in chunks:
                        if chunk.strip():
                            texts.append(chunk.strip())
        print(f"✅ Extracted {len(texts)} chunks from PDF.")
    except Exception as e:
        print(f"❌ PDF error: {e}")
    return texts

# -----------------------------
# 12. Extract Text from Word Doc
# -----------------------------
def extract_from_word(filepath):
    texts = []
    try:
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            if para.text.strip():
                texts.append(para.text.strip())
        print(f"✅ Extracted {len(texts)} paragraphs from Word doc.")
    except Exception as e:
        print(f"❌ Word doc error: {e}")
    return texts

# -----------------------------
# 13. Import from File
# -----------------------------
def import_from_file():
    filepath = input("Enter file path (PDF or .docx): ").strip()

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    if filepath.lower().endswith(".pdf"):
        texts = extract_from_pdf(filepath)
    elif filepath.lower().endswith(".docx"):
        texts = extract_from_word(filepath)
    else:
        print("❌ Only PDF and .docx files are supported.")
        return

    if not texts:
        print("❌ No text extracted from file.")
        return

    docs = load_documents()
    embeddings = load_embeddings()
    existing_texts = {e["text"] for e in embeddings}
    added = 0

    for text in texts:
        if text not in docs:
            docs.append(text)
            if text not in existing_texts:
                print(f"🔄 Embedding: {text[:60]}...")
                emb = get_embedding(text)
                embeddings.append({"text": text, "embedding": emb})
                added += 1

    save_documents(docs)
    save_embeddings(embeddings)
    print(f"✅ Imported and embedded {added} new chunks!")

# -----------------------------
# 14. Main Menu
# -----------------------------
def main():
    while True:
        print("\n" + "="*45)
        print("   🔍 Semantic Search — AWS Bedrock")
        print("="*45)
        print("  1. Create / Refresh Embeddings")
        print("  2. Search")
        print("  3. Add New Document")
        print("  4. Delete Document")
        print("  5. List All Documents")
        print("  6. Import from PDF / Word Doc")
        print("  0. Exit")
        print("="*45)

        choice = input("Choose: ").strip()

        if choice == "1":
            create_embeddings()
        elif choice == "2":
            query = input("Enter your search query: ").strip()
            if not query:
                print("❌ Query cannot be empty.")
                continue
            results = search(query)
            if not results:
                print("⚠️  No results above similarity threshold.")
            else:
                print("\n🔎 Top Results:")
                for text, score in results:
                    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                    print(f"\n  📄 {text[:80]}")
                    print(f"     {bar} {score:.4f}")
        elif choice == "3":
            text = input("Enter new document text: ").strip()
            add_document(text)
        elif choice == "4":
            delete_document()
        elif choice == "5":
            list_documents()
        elif choice == "6":
            import_from_file()
        elif choice == "0":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Choose 0-6.")

if __name__ == "__main__":
    main()