import os
import lancedb
import openai
import re # Import the regular expressions library
from dotenv import load_dotenv
from lancedb.pydantic import LanceModel, Vector

# --- CONFIGURATION ---

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
db = lancedb.connect("./support_db")

# --- UPDATED DATA MODEL ---
class SupportDoc(LanceModel):
    text: str
    vector: Vector(1536)
    source_url: str

# Create or open the table
try:
    table = db.open_table("support_docs")
    print("LanceDB table 'support_docs' opened for ingestion.")
except Exception:
    table = db.create_table("support_docs", schema=SupportDoc)
    print("LanceDB table 'support_docs' created for ingestion.")

# --- INGESTION LOGIC ---

def chunk_text_semantically(text: str):
    """Splits text into chunks based on markdown headers."""
    # We use a positive lookahead regex to split the text on markdown headers
    # while keeping the header with its content.
    chunks = re.split(r'(?=\n#\s)', text)
    # Filter out any empty chunks that might result from the split
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def ingest_folder(folder_path):
    """Reads all .txt files from a folder, extracts source URLs, and adds them to the database."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"Read {len(content)} characters from {filename}")
                
                # --- NEW: Extract source URL ---
                source_url = "No URL provided"
                if content.startswith("source_url:"):
                    lines = content.split('\n')
                    source_url = lines[0].replace("source_url:", "").strip()
                    content = "\n".join(lines[1:]) # Remove the URL line from the content
                    print(f"Found source URL: {source_url}")

                chunks = chunk_text_semantically(content)
                print(f"Split into {len(chunks)} semantic chunks.")
                
                # --- NEW: Associate URL with each chunk ---
                for chunk in chunks:
                    documents.append({"text": chunk, "source_url": source_url})

    if not documents:
        print("No documents to ingest.")
        return

    print("Creating embeddings for all chunks... This may take a while.")
    # Create embeddings for all chunks at once
    response = openai.embeddings.create(
        input=[doc["text"] for doc in documents],
        model="text-embedding-ada-002"
    )
    embeddings = [item.embedding for item in response.data]

    # Prepare data for insertion
    data_to_add = [
        {"text": doc["text"], "vector": emb, "source_url": doc["source_url"]} 
        for doc, emb in zip(documents, embeddings)
    ]

    # Add the data to the table
    table.add(data_to_add)
    print(f"Successfully ingested {len(data_to_add)} chunks into the database.")

if __name__ == "__main__":
    knowledge_folder = "./knowledge"
    # IMPORTANT: Delete the old database before running, to avoid duplicates
    import shutil
    db_path = "./support_db"
    if os.path.exists(db_path):
        print(f"Deleting old database at {db_path}...")
        shutil.rmtree(db_path)
    
    ingest_folder(knowledge_folder)