import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from typing import Optional, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from mcp.types import Tool
import os
import logging
from rank_bm25 import BM25Okapi
from host import MCPHost, console
from rich.table import Table
logging.basicConfig(level=logging.WARNING)
load_dotenv()  # load environment variables from .env

class ToolRetrieval:
    """Manages BM25-based tool retrieval"""
    
    def __init__(self):
        self.bm25 = None
        self.tool_names = []
        self.tool_metadata = {}
        self.documents = []
    
    def index_tools(self, tools: List[Tool]):
        """Create BM25 index for tools"""
        self.tool_names = []
        self.documents = []
        self.tool_metadata = {}
        
        for tool in tools:
            # Create rich document for BM25
            doc_content = f"{tool.name} {tool.description}"
            
            # Add parameter names and descriptions
            if "properties" in tool.inputSchema:
                props = tool.inputSchema["properties"]
                for param_name, param_info in props.items():
                    param_desc = param_info.get('description', '')
                    doc_content += f" {param_name} {param_desc}"
            
            # Tokenize document
            tokens = doc_content.lower().split()
            self.documents.append(tokens)
            self.tool_names.append(tool.name)
            self.tool_metadata[tool.name] = {
                "description": tool.description,
                "input_schema": tool.inputSchema,
                "original_tool": tool
            }
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.documents)
        console.print(f"[dim]BM25 index created for [bold]{len(self.tool_names)}[/] tools[/]")
    
    def retrieve_tools(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k relevant tools using BM25"""
        if self.bm25 is None:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Rank tools by score
        ranked_tools = sorted(
            zip(self.tool_names, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-k tool names with scores
        top_tools = ranked_tools[:top_k]
        retrieved_tools = [tool[0] for tool in top_tools if tool[1] > 0]

        table = Table(title="BM25 Retrieval Scores", show_header=True, header_style="bold magenta")
        table.add_column("Tool", style="green")
        table.add_column("Score", justify="right", style="cyan")
        for name, score in top_tools:
            table.add_row(name, f"{score:.3f}")
        console.print(table)

        return retrieved_tools if retrieved_tools else [self.tool_names[0]] if self.tool_names else []


class MCPRAGHost(MCPHost):
    def __init__(self):
        super().__init__()
        self.tool_retrieval = None
        
    async def connect_to_server(self, server_script_path: str):
        """Connect to the MCP server and index tools for RAG-based retrieval"""
        await super().connect_to_server(server_script_path)
        self.tool_retrieval = ToolRetrieval()
        self.tool_retrieval.index_tools(self.tools)
        console.print("[bold green]RAG-based tool selection ready.[/]")
    
    def _build_context_from_history(self, query: str, messages: List[dict]) -> str:
        """Build a rich search context from conversation history and current query"""
        context_parts = []
        
        # Add recent conversation context (last 4 messages)
        recent_messages = messages[-4:] if len(messages) > 4 else messages
        for msg in recent_messages:
            if msg.get("role") in ["user", "assistant"]:
                content = msg.get("content", "")
                if content:
                    context_parts.append(content)
        
        # Add current query
        context_parts.append(query)
        
        # Combine all context
        full_context = " ".join(context_parts)
        console.print(f"[dim]Search context:[/] {full_context[:200]}...")

        return full_context
    
    def get_tools_description(self, query: str, messages: List[dict]) -> str:
        # Describe available tools to the model and include the user query

        search_context = self._build_context_from_history(query, messages)
        
        # Use BM25 to retrieve relevant tools
        relevant_tool_names = self.tool_retrieval.retrieve_tools(search_context, top_k=3)
        console.print(f"[bold yellow]Retrieved tools via BM25:[/] {relevant_tool_names}")
        
        # Get full tool objects
        all_tools = self.tools
        selected_tools = [t for t in all_tools if t.name in relevant_tool_names]
        
        # Build system prompt with only selected tools
        tools_description = "\n".join([self.format_for_llm(tool) for tool in selected_tools])
        return tools_description


async def main():
    client = MCPRAGHost()
    try:
        await client.connect_to_server("http://localhost:8000")  # Connect to the MCP server
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
    
# uv run host-rag.py ../mcp-server/weather.py