from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import os
from src.config import RAGConfig

class IndexingEngine:
    def __init__(self, data_dir= "data", persist_dir= "storage"):
        # Ensure Config is setup
        RAGConfig.setup()

        self.data_dir= data_dir
        self.persist_dir= persist_dir
        self.index= None

    def build_or_load(self):
        """Checks if index exists on disk. If yes, loads it. If no, builds it."""
        if os.path.exists(self.persist_dir):
            print(f"Loading existing Index from {self.persist_dir}...")
            storage_context= StorageContext.from_defaults(persist_dir= self.persist_dir)
            self.index= load_index_from_storage(storage_context)
        else:
            print(f"Building Fresh Index from {self.data_dir}...")

            # Load Documents
            documents= SimpleDirectoryReader(self.data_dir).load_data()

            #Create Index
            self.index= VectorStoreIndex.from_documents(documents)

            # Save It!
            self.index.storage_context.persist(persist_dir= self.persist_dir)
            print(f"Index Saved to {self.persist_dir}")

        return self.index

if __name__ == "__main__":
    engine= IndexingEngine()
    index= engine.build_or_load()
    print("Indexing Engine Ready!")