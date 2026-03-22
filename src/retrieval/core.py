from llama_index.core import Settings
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from src.indexing.engine import IndexingEngine
from src.config import RAGConfig

class AdvancedRetriever:
    def __init__(self):
        # 1. Setup Config & Index
        self.engine= IndexingEngine()
        self.index= self.engine.build_or_load()

        # 2. Define the Base Retriever
        self.base_retriever= self.index.as_retriever(similarity_top_k= 10)

        # 3. Define the Reranker (The Quality Filter)
        self.reranker= LLMRerank(
            choice_batch_size= 5,
            top_n= 3,
        )

    def get_query_engine(self):
        """Returns a Query Engine with Reranking built-in."""
        return RetrieverQueryEngine.from_args(
            retriever= self.base_retriever,
            node_postprocessors= [self.reranker],
            verbose= True
        )

if __name__ == "__main__":
    retrieval_system= AdvancedRetriever()
    query_engine= retrieval_system.get_query_engine()

    response= query_engine.query("What is DevGuardian?")
    print(f"\nTEST RETRIEVAL RESPONSE\n {response}")