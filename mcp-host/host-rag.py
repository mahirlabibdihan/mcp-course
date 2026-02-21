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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
        logging.info(f"BM25 index created for {len(self.tool_names)} tools")
    
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
        
        logging.info(f"BM25 retrieval scores: {[(t[0], round(t[1], 3)) for t in top_tools]}")
        
        return retrieved_tools if retrieved_tools else [self.tool_names[0]] if self.tool_names else []


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Initialize OpenAI client (supports OpenRouter via base_url)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openai_api_key,
        )
        self.model = "openai/gpt-5-mini"
        self.servers = []
        self.tools = []
        self.tool_retrieval = None

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json
        
        def _clean_json_string(json_string: str) -> str:
            """Remove ```json ... ``` or ``` ... ``` wrappers if the LLM response is fenced."""
            import re

            pattern = r"^```(?:\s*json)?\s*(.*?)\s*```$"
            return re.sub(pattern, r"\1", json_string, flags=re.DOTALL | re.IGNORECASE).strip()

        final_text = []
        try:
            tool_call = json.loads(_clean_json_string(llm_response))
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")

                response = await self.session.list_tools()
                tools = response.tools
                if any(tool.name == tool_call["tool"] for tool in tools):
                    try:
                        result = await self.session.call_tool(tool_call["tool"], tool_call["arguments"])
                        final_text.append(f"[Calling tool {tool_call['tool']} with args {tool_call['arguments']}]")

                        if isinstance(result, dict) and "progress" in result:
                            progress = result["progress"]
                            total = result["total"]
                            percentage = (progress / total) * 100
                            logging.info(f"Progress: {progress}/{total} ({percentage:.1f}%)")

                        logging.info(f"\n> Tool execution result: \n{result.content[0].text}")
                        return f"Tool execution result: {result.content[0].text}"
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        logging.error(error_msg)
                        return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response
        
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        if is_python:
            path = Path(server_script_path).resolve()
            server_params = StdioServerParameters(
                command="uv",
                args=["--directory", str(path.parent), "run", path.name],
                env=None,
            )
        else:
            server_params = StdioServerParameters(
                command="node", args=[server_script_path], env=None
            )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        
        self.tool_retrieval = ToolRetrieval()
        self.tool_retrieval.index_tools(tools)
        print("BM25 index created for RAG-based tool selection")

    def format_for_llm(self, tool: Tool) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        # print(tool)
        if "properties" in tool.inputSchema:
            for param_name, param_info in tool.inputSchema["properties"].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in tool.inputSchema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        # Build the formatted output with title as a separate field
        output = f"Tool: {tool.name}\n"

        output += f"""Description: {tool.description}
Arguments:
{chr(10).join(args_desc)}
"""
        return output
    
    def extract_text(self, resp):
        # Extract text from response (support dict-like or attribute access)
        text = ""
        try:
            # resp.choices[0].message.content (common shape)
            if hasattr(resp, 'choices') and len(resp.choices) > 0:
                choice = resp.choices[0]
                # choice may be a dict-like or object
                if isinstance(choice, dict):
                    msg = choice.get('message') or choice.get('delta') or {}
                    if isinstance(msg, dict):
                        text = msg.get('content') or msg.get('role') or str(msg)
                    else:
                        text = str(msg)
                else:
                    # attribute access
                    msg = getattr(choice, 'message', None)
                    if msg is not None:
                        text = getattr(msg, 'content', str(msg))
                    else:
                        text = str(choice)
            else:
                text = str(resp)
        except Exception:
            text = str(resp)
        return text
    
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
        logging.info(f"Search context: {full_context[:200]}...")
        
        return full_context
    
    
    async def process_query(self, query: str, messages) -> str:
        """Process a query using Claude and available tools"""
        # Describe available tools to the model and include the user query

        search_context = self._build_context_from_history(query, messages)
        
        # Use BM25 to retrieve relevant tools
        relevant_tool_names = self.tool_retrieval.retrieve_tools(search_context, top_k=3)
        logging.info(f"Retrieved tools via BM25: {relevant_tool_names}")
        
        # Get full tool objects
        response = await self.session.list_tools()
        all_tools = response.tools
        selected_tools = [t for t in all_tools if t.name in relevant_tool_names]
        
        # Build system prompt with only selected tools
        tools_description = "\n".join([self.format_for_llm(tool) for tool in selected_tools])
        
        system_message = (
            "You are a helpful assistant with access to these tools:\n\n"
            f"{tools_description}\n"
            "Choose the appropriate tool based on the user's question. "
            "If no tool is needed, reply directly.\n\n"
            "IMPORTANT: When you need to use a tool, you must ONLY respond with "
            "the exact JSON object format below, nothing else:\n"
            "{\n"
            '    "tool": "tool-name",\n'
            '    "arguments": {\n'
            '        "argument-name": "value"\n'
            "    }\n"
            "}\n\n"
            "After receiving a tool's response:\n"
            "1. Transform the raw data into a natural, conversational response\n"
            "2. Keep responses concise but informative\n"
            "3. Focus on the most relevant information\n"
            "4. Use appropriate context from the user's question\n"
            "5. Avoid simply repeating the raw data\n\n"
            "Please use only the tools that are explicitly defined above."
        )
        
        logging.info(f"\n# System Prompt: \n{system_message}")
        
        messages.append({"role": "user", "content": query})

        # Call OpenAI synchronously in a thread to avoid blocking the event loop
        def _call_openai():
            return self.extract_text(self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    *messages,
                ],
                max_tokens=1000,
                # tools=available_tools,
            ))

        text = await asyncio.to_thread(_call_openai)
        logging.info("\n> Assistant: %s", text)
        
        result = await self.process_llm_response(text)
        if result != text:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "system", "content": result})
            final_response = await asyncio.to_thread(_call_openai)
            logging.info("\n> Final response: %s", final_response)
            messages.append({"role": "assistant", "content": final_response})
        else:
            messages.append({"role": "assistant", "content": text})
        return result

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        messages = []
        while True:
            try:
                query = input("\n> Query: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query, messages)
                # print("\n" + response)
            except Exception as e:
                print(f"\nError: {e}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())