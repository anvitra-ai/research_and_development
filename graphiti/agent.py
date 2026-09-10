from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
import asyncio
import os

from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from pydantic_ai import Agent, RunContext
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
import graphiti_core.helpers as graphiti_helpers

# Max concurrent LLM/DB operations (must be set before graphiti_core is imported)
CONCURRENCY_LIMIT = 50
os.environ["SEMAPHORE_LIMIT"] = str(CONCURRENCY_LIMIT)

load_dotenv()

# ========== Define dependencies ==========
@dataclass
class GraphitiDependencies:
    """Dependencies for the Graphiti agent."""
    graphiti_client: Graphiti

# ========== Helper function to get model configuration ==========
def get_model():
    """Configure and return the Gemini model used by Pydantic AI."""

    model_choice = os.getenv("MODEL_CHOICE", "gemini-3.6-flash")
    api_key = os.getenv("GEMINI_API_KEY","****")

    return GoogleModel(
        model_choice,
        provider=GoogleProvider(api_key=api_key),
    )

# ========== Create the Graphiti agent ==========
graphiti_agent = Agent(
    get_model(),
    system_prompt="""You are a helpful assistant with access to a knowledge graph filled with temporal data about LLMs.
    When the user asks you a question, use your search tool to query the knowledge graph and then answer honestly.
    Be willing to admit when you didn't find the information necessary to answer the question.""",
    deps_type=GraphitiDependencies
)

# ========== Define a result model for Graphiti search ==========
class GraphitiSearchResult(BaseModel):
    """Model representing a search result from Graphiti."""
    uuid: str = Field(description="The unique identifier for this fact")
    fact: str = Field(description="The factual statement retrieved from the knowledge graph")
    valid_at: Optional[str] = Field(None, description="When this fact became valid (if known)")
    invalid_at: Optional[str] = Field(None, description="When this fact became invalid (if known)")
    source_node_uuid: Optional[str] = Field(None, description="UUID of the source node")

# ========== Graphiti search tool ==========
@graphiti_agent.tool
async def search_graphiti(ctx: RunContext[GraphitiDependencies], query: str) -> List[GraphitiSearchResult]:
    """Search the Graphiti knowledge graph with the given query.
    
    Args:
        ctx: The run context containing dependencies
        query: The search query to find information in the knowledge graph
        
    Returns:
        A list of search results containing facts that match the query
    """
    # Access the Graphiti client from dependencies
    graphiti = ctx.deps.graphiti_client
    
    try:
        # Perform the search
        results = await graphiti.search(query)
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = GraphitiSearchResult(
                uuid=result.uuid,
                fact=result.fact,
                source_node_uuid=result.source_node_uuid if hasattr(result, 'source_node_uuid') else None
            )
            
            # Add temporal information if available
            if hasattr(result, 'valid_at') and result.valid_at:
                formatted_result.valid_at = str(result.valid_at)
            if hasattr(result, 'invalid_at') and result.invalid_at:
                formatted_result.invalid_at = str(result.invalid_at)
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    except Exception as e:
        # Log the error but don't close the connection since it's managed by the dependency
        print(f"Error searching Graphiti: {str(e)}")
        raise

# ========== Main execution function ==========
async def main():
    """Run the Graphiti agent with user queries."""
    print("Graphiti Agent - Powered by Pydantic AI, Graphiti, and Neo4j")
    print("Enter 'exit' to quit the program.")

    # Neo4j connection parameters
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', '****')
    
    # Initialize Graphiti with Neo4j connection
    # Apply concurrency limit (also updates module if graphiti_core was imported earlier)

    # Google API key configuration
    api_key = "***"

    # Initialize Graphiti with Gemini clients
    graphiti_client = Graphiti(
    "bolt://127.0.0.1:7687",
    "neo4j",
    "****",

    llm_client=GeminiClient(
        config=LLMConfig(
            api_key=api_key,
            model="gemini-3.1-pro-preview",      # Best reasoning model
            small_model="gemini-3.6-flash",      # Fast production model
        )
    ),

    embedder=GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=api_key,
            embedding_model="gemini-embedding-001",  # Latest embedding model
            #embedding_dim=1024                     # Verify this matches your wrapper's expected dimension
        )
    ),

    cross_encoder=GeminiRerankerClient(
        config=LLMConfig(
            api_key=api_key,
            model="gemini-3.6-flash",
            small_model="gemini-3.6-flash",
        )
    ),

    max_coroutines=CONCURRENCY_LIMIT,
)    
    # Initialize the graph database with graphiti's indices if needed
    try:
        await graphiti_client.build_indices_and_constraints()
        print("Graphiti indices built successfully.")
    except Exception as e:
        print(f"Note: {str(e)}")
        print("Continuing with existing indices...")

    console = Console()
    messages = []
    
    try:
        while True:
            # Get user input
            user_input = input("\n[You] ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Goodbye!")
                break
            
            try:
                # Process the user input and output the response
                print("\n[Assistant]")
                with Live('', console=console, vertical_overflow='visible') as live:
                    # Pass the Graphiti client as a dependency
                    deps = GraphitiDependencies(graphiti_client=graphiti_client)
                    
                    async with graphiti_agent.run_stream(
                        user_input, message_history=messages, deps=deps
                    ) as result:
                        curr_message = ""
                        async for message in result.stream_text(delta=True):
                            curr_message += message
                            live.update(Markdown(curr_message))
                    
                    # Add the new messages to the chat history
                    messages.extend(result.all_messages())
                
            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")
    finally:
        # Close the Graphiti connection when done
        await graphiti_client.close()
        print("\nGraphiti connection closed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise