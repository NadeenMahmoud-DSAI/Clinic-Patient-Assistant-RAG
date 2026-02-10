import os
import re
import numpy as np
import faiss
import requests
import json
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

class PDFVectorDatabase:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the PDF Vector Database with embedding model."""
        print(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.dimension = 384  
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def create_chunks(self, text: str, chunk_size: int = 100, overlap: int = 50) -> List[str]:
        
        sentences = re.split(r'(?<=[.?!])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            len_sentence = len(sentence_words)
            
            if word_count + len_sentence > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                current_words = current_chunk.split()
                if len(current_words) > overlap:
                    overlap_text = ' '.join(current_words[-overlap:])
                    current_chunk = overlap_text + " " + sentence
                    word_count = overlap + len_sentence
                else:
                    current_chunk = sentence
                    word_count = len_sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                word_count += len_sentence
        
        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_faiss_database(self, pdf_path: str, chunk_size: int = 150, overlap: int = 50):
        """Create FAISS vector database from PDF."""
        print("Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            print("No text extracted from PDF.")
            return
        
        print("Creating chunks...")
        self.chunks = self.create_chunks(text, chunk_size, overlap)
        print(f"Created {len(self.chunks)} chunks.")
        
        print("Creating embeddings (this may take a moment)...")
        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        print("Creating FAISS index...")
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        print(f"Database ready with {self.index.ntotal} vectors.")
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Perform semantic search on the FAISS database."""
        if self.index is None:
            print("No FAISS index found. Please create the database first.")
            return []
        
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(distance)))
        
        return results

    def query_with_openrouter(self, question: str, api_key: str, model: str = "openrouter/auto", top_k: int = 3) -> str:       
        """Query the LLM using OpenRouter with retrieved context."""
        
        # 1. Retrieve Context
        print(f"Retrieving context for: '{question}'")
        search_results = self.semantic_search(question, top_k=top_k)
        
        if not search_results:
            return "No relevant context found in the database."
        
        context_str = "\n\n".join([f"[Context {i+1}]: {chunk}" for i, (chunk, _) in enumerate(search_results)])
        
        # System Prompt
        system_prompt = f"""
        Role: You are the virtual patient services assistant for HealthFirst Clinic.
        Instructions: Use the provided context to answer the user's question.
        - If the answer is in the context, answer clearly.
        - If the answer is NOT in the context, state: "I don't have that information. Please call (555) 0199."
        - Do not make up facts not present in the context.
        
        --- BEGIN CONTEXT ---
        {context_str}
        --- END CONTEXT ---
        """

        # API Request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "ClinicBot"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        }
        
        # 5. Execute Request
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                return f"Error: OpenRouter API returned {response.status_code} - {response.text}"
        except Exception as e:
            return f"Exception during API call: {e}"

if __name__ == "__main__":
    # Configuration
    PDF_FILE = "clinic_policies.pdf"
    API_KEY = "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"  
    
    if not os.path.exists(PDF_FILE):
        print(f"Error: {PDF_FILE} not found. Please place the file in the same directory.")
    else:
        # Initialize Database
        db = PDFVectorDatabase()
        db.create_faiss_database(PDF_FILE)
        
        # Interactive Loop
        print("\n--- HealthFirst Clinic Assistant (Type 'quit' to exit) ---")
        while True:
            user_input = input("\nPatient: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            # Answer
            answer = db.query_with_openrouter(user_input, api_key=API_KEY)
            print(f"Assistant: {answer}")