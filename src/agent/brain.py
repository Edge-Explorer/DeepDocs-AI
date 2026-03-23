from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from src.retrieval.core import AdvancedRetriever
from src.config import RAGConfig 

class RAGAgent:
    def __init__(self):
        # 1. Setup Config
        RAGConfig.setup()

        # 2. Setup the Retrieval System
        self.retrieval_system= AdvancedRetriever()
        self.query_engine= self.retrieval_system.get_query_engine()

        # 3. Define the Tool
        self.tools= [
            QueryEngineTool(
                query_engine= self.query_engine,
                metadata= ToolMetadata(
                    name= "dev_guardian_docs",
                    description= "Official documentation for DevGuardian security tools and features. Use this for specific technical questions about the project."
                )
            )
        ]

        # 4. Initialize the ReAct Agent
        self.agent= ReActAgent(
            tools= self.tools,
            verbose= True,
        )
    async def chat(self, user_msg: str):
        """Asynchronously chat with the agent."""
        # Using .run() or .chat() depending on your version
        # Note: If your version uses Workflow, use .run(user_msg=...)
        response = await self.agent.run(user_msg=user_msg)
        return response
        
if __name__ == "__main__":
    import asyncio
    
    async def test():
        brain = RAGAgent()
        print("BRAIN ACTIVE: I can now reason and search.")
        
        # Ask a complex question
        response = await brain.chat("Briefly summarize what makes DevGuardian different from a standard RAG.")
        print(f"\n--- FINAL AGENT RESPONSE ---\n{response}")
    asyncio.run(test())