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
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
logging.basicConfig(level=logging.WARNING)
console = Console()
load_dotenv()  # load environment variables from .env


class MCPHost:
    def __init__(self):
        # Initialize server and client objects
        self.client: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Initialize OpenAI client (supports OpenRouter via base_url)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openai_api_key,
        )
        self.model = "openai/gpt-5-mini"
        self.tools = []

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
                console.print(f"[bold yellow]Executing tool:[/] [green]{tool_call['tool']}[/]")
                console.print(f"[dim]Arguments:[/] {tool_call['arguments']}")

                if any(tool.name == tool_call["tool"] for tool in self.tools):
                    try:
                        result = await self.client.call_tool(tool_call["tool"], tool_call["arguments"])
                        final_text.append(f"[Calling tool {tool_call['tool']} with args {tool_call['arguments']}]")

                        console.print(Panel(
                            result.content[0].text,
                            title="[bold cyan]Tool Result[/]",
                            border_style="cyan",
                        ))
                        return f"Tool execution result: {result.content[0].text}"
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        console.print(f"[bold red]Error executing tool:[/] {error_msg}")
                        return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response
        
    async def connect_to_server(self, server_url: str):
        """Connect to an MCP server
        
        Args:
            server_url: The URL of the MCP server (e.g. http://localhost:8000)
        """

        read, write, _ = await self.exit_stack.enter_async_context(
            streamablehttp_client(server_url + "/mcp")
        )

        self.client = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self.client.initialize()
        
        # List available tools
        response = await self.client.list_tools()
        tools = response.tools
        self.tools = tools
        tool_names = ", ".join(f"[green]{t.name}[/]" for t in tools)
        console.print(Panel(
            f"[bold]Available tools:[/] {tool_names}",
            title="[bold green]Connected to MCP Server[/]",
            border_style="green",
        ))

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
    
    def get_tools_description(self, query: str, messages: List[dict]) -> str:
        # Describe available tools to the model and include the user query
        tools_description = "\n".join([self.format_for_llm(tool) for tool in self.tools])
        return tools_description
    
    def get_system_prompt(self, query: str, messages: List[dict]) -> str:
        tools_description = self.get_tools_description(query, messages)

        system_prompt = (
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
        
        return system_prompt
        
    async def process_query(self, query: str, messages) -> str:
        """Process a query using Claude and available tools"""
        
        system_prompt = self.get_system_prompt(query, messages)
        # logging.info(f"\n# System Prompt: \n{system_prompt}")
        
        # Describe available tools to the model and include the user query
        messages.append({"role": "user", "content": query})
        import json
        
        # Call OpenAI synchronously in a thread to avoid blocking the event loop
        def _call_openai():
            return self.extract_text(self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages,
                ],
                max_tokens=1000,
                # tools=available_tools,
            ))

        text = await asyncio.to_thread(_call_openai)
        console.print(f"[dim]Assistant (raw):[/] {text}")
        
        result = await self.process_llm_response(text)
        if result != text:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "system", "content": result})
            final_response = await asyncio.to_thread(_call_openai)
            console.print(Panel(
                Markdown(final_response),
                title="[bold blue]Assistant[/]",
                border_style="blue",
            ))
            messages.append({"role": "assistant", "content": final_response})
        else:
            console.print(Panel(
                Markdown(text),
                title="[bold blue]Assistant[/]",
                border_style="blue",
            ))
            messages.append({"role": "assistant", "content": text})
        return result

    async def chat_loop(self):
        """Run an interactive chat loop"""
        console.print(Panel(
            "[bold]Type your queries or [red]quit[/red] to exit.[/]",
            title="[bold magenta]MCP Host Started[/]",
            border_style="magenta",
        ))

        messages = []
        while True:
            try:
                console.print(Rule(style="dim"))
                query = console.input("[bold cyan]Query:[/] ").strip()
                if query.lower() == 'quit':
                    console.print("[dim]Goodbye![/]")
                    break
                response = await self.process_query(query, messages)
            except Exception as e:
                console.print(f"[bold red]Error:[/] {e}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    client = MCPHost()
    try:
        await client.connect_to_server("http://localhost:8000")
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
    
# uv run host.py ../mcp-server/weather.py