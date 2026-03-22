import os
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings

load_dotenv()

class RAGConfig:
    # Model Names
    LLM_MODEL= "models/gemini-2.0-flash"
    EMBED_MODEL= "models/gemini-embedding-2-preview"

    #RAG Settings
    CHUNK_SIZE= 512
    CHUNK_OVERLAP= 50
    SIMILARITY_TOP_K= 10
    RERANK_TOP_N= 3

    @staticmethod
    def setup():
        """Initializes global LlamaIndex settings"""
        api_key= os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        Settings.llm = GoogleGenAI(model=RAGConfig.LLM_MODEL, api_key=api_key)
        Settings.embed_model = GoogleGenAIEmbedding(model_name=RAGConfig.EMBED_MODEL, api_key=api_key)
        Settings.chunk_size = RAGConfig.CHUNK_SIZE
        Settings.chunk_overlap = RAGConfig.CHUNK_OVERLAP
        
        print(f"RAG Configized with {RAGConfig.LLM_MODEL}")

if __name__ == "__main__":
    RAGConfig.setup()
    print(f"LLM Model: {Settings.llm.model}")
    print(f"Embed Model: {Settings.embed_model.model_name}")