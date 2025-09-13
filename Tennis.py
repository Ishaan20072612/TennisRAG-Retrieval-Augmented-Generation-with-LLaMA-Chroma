# ============================================================
# 1. Install Dependencies
# =============================

# !pip install langchain_community langchain_text_splitters chromadb sentence-transformers transformers huggingface_hub
import os

# =============================
# 2. Load the Dataset
# =============================

from langchain.document_loaders import TextLoader
loader = TextLoader("/content/tennis_details.md")
text_doc = loader.load()
print("🔹 Raw Document Object:\n", text_doc)
print("\n🔹 First Page Content:\n", text_doc[0].page_content)

# =============================
# 4. Split Document into Chunks
# =============================

from langchain_text_splitters import MarkdownHeaderTextSplitter
split_condition = [("##", "Title")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=split_condition)
doc_splits = splitter.split_text(text_doc[0].page_content)
print("\n🔹 Split Documents:\n", doc_splits)
text_chunks = [i.page_content for i in doc_splits]
print("\n🔹 Text Chunks:\n", text_chunks)

# =============================
# 5. Generate Embeddings
# =============================

from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunk(chunk):
    """Generate embeddings for a given chunk of text"""
    return embedding_model.encode([chunk], normalize_embeddings=True)

# Example: Embedding of the 7th chunk
sample_embeddings = embed_chunk(text_chunks[6])
print("\n🔹 Sample Embeddings:\n", sample_embeddings)

# =============================
# 6. Create Vector Database
# =============================

from langchain.vectorstores import Chroma
vector_db = Chroma.from_texts(
    text_chunks,
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    persist_directory="/tmp/chromadb/"
)

print("\n🔹 Stored Vectors:\n", vector_db._collection.get(include=['embeddings', 'documents']))

# =============================
# 7. Setup LLM (Text Generation)
# =============================

from huggingface_hub import login
login()  # Requires HuggingFace token

from transformers import pipeline
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
print("\n🔹 Pipeline Loaded:\n", pipe)

# =============================
# 8. Retrieval + Generation Function
# =============================

def retrieve_and_generate(query, threshold=1):
    """Retrieve context from VectorDB and generate answer with LLM"""
    
    search_results = vector_db.similarity_search_with_score(query, k=1)
    
    if not search_results or search_results[0][1] > threshold:
        return "I don't know the answer"
    
    retrieve_context = search_results[0][0].page_content
    similarity_score = search_results[0][1]
    
    print(f"\n🔹 Similarity Score: {similarity_score}")
    print(f"🔹 Retrieved Context: {retrieve_context}")
    
    prompt = f"""
    Answer the question using the given context.
    Context: {retrieve_context}
    Question: {query}
    Answer:
    """
    
    response = pipe(prompt, max_new_tokens=100)
    return response[0]["generated_text"]

# =============================
# 9. Test the RAG Pipeline
# =============================

ques1 = "What is tennis?"
response1 = retrieve_and_generate(ques1)
print("\n❓ Q:", ques1)
print("✅ A:", response1)

ques2 = "What is gated models?"
response2 = retrieve_and_generate(ques2)
print("\n❓ Q:", ques2)
print("✅ A:", response2)