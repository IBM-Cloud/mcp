# The Python program acts as an agent, using an LLM from watsonx.ai to orchestrate tools managed by an MCP

# --- ASSUMPTIONS & CAUTIONS ---

# This program, designed to integrate with watsonx.ai for LLM capabilities and an MCP (Microservice Control Plane) server for tool orchestration, makes several assumptions about LLM behavior and requires careful consideration due to its dependency on the LLM.

# Assumptions Related to LLM Returns
    # Shortlisting Tools
        # LLM's Ability to Select Tools: The program assumes the LLM, given a list of available tools and a user query, can accurately identify which tool(s) are relevant and should be called. It relies on the LLM to understand the descriptions and parameters of the tools provided to it.

        # Prompt Engineering for Tool Calling: It's assumed that the specific prompt engineering used to present tools to the LLM will reliably elicit a response in the expected JSON format for tool calls (similar to OpenAI's tool_calls structure). If the LLM deviates from this format, the program's parsing logic will fail.

    # Function Returns (Boolean as String or Vice Versa)
        # Data Type Consistency: The comments highlight a specific concern: "sometimes for functions it may return boolean as string or viceversion." This implies an assumption that the LLM might not always return data in the expected native Python data types (e.g., a boolean True might be returned as the string "True"). The program's subsequent logic for using these values as arguments in tool calls or for processing tool outputs must be robust enough to handle such type inconsistencies through explicit type casting or validation.

# Note: I've conducted initial tests, verifying the basic functionalities of the ibmcloud_resource_groups for listing resource groups, ibmcloud_cos_buckets for listing COS buckets, and ibmcloud_cos_config_crn and ibmcloud_cos_config_region for configuring COS service details.

# --- ASSUMPTIONS & CAUTIONS ---

import asyncio
import json
import os
from fastmcp import Client

# Import necessary classes from watsonx.ai SDK
# from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.credentials import Credentials


# --- Configuration ---
MCP_SERVER_URL = "http://localhost:8000/sse" # Adjust this as needed

# --- IMPORTANT: Configure your watsonx.ai details ---
# You MUST replace these with your actual watsonx.ai credentials.
# It's highly recommended to load these from environment variables for security.
WX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WX_API_URL = os.getenv("WATSONX_API_URL", "") # Adjust if your region is different
WX_MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503" # Example model, choose one that supports chat/tooling

# --- MCP Client Class ---
class MCPClient:
    def __init__(self, mcp_server_url: str, wx_api_key: str, wx_project_id: str, wx_api_url: str, wx_model_id: str):
        self.mcp_server_url = mcp_server_url
        self.wx_api_key = wx_api_key
        self.wx_project_id = wx_project_id
        self.wx_api_url = wx_api_url
        self.wx_model_id = wx_model_id
        self.conversation_history = []
        
        # Initialize the fastmcp client
        self.mcp_client = Client(self.mcp_server_url)

        # Initialize watsonx.ai model
        self.credentials = Credentials(
            api_key=self.wx_api_key,
            url=self.wx_api_url
        )
        # Default generation parameters for the LLM
        # Adjust these as needed for your model and desired behavior
        self.generate_params = {
            GenTextParamsMetaNames.MAX_NEW_TOKENS: 500,
            GenTextParamsMetaNames.TEMPERATURE: 0.7,
            GenTextParamsMetaNames.DECODING_METHOD: DecodingMethods.SAMPLE,
            GenTextParamsMetaNames.STOP_SEQUENCES: [] # Important for tool calling: LLMs might generate specific stop sequences before tool calls
        }

        self.llm_model = ModelInference(
            model_id=self.wx_model_id,
            credentials=self.credentials,
            project_id=self.wx_project_id,
            params=self.generate_params
        )

    async def _discover_tools(self) -> list:
        """
        Calls the MCP server to get the list of tools using fastmcp.Client.
        Returns a list of dictionaries, suitable for LLM tool schema.
        """
        print(f"DEBUG: Discovering tools from MCP server at {self.mcp_server_url}...")
        try:
            tools_data = await self.mcp_client.list_tools()
            
            llm_compatible_tools = []
            for tool_def in tools_data:
                if tool_def.inputSchema:
                    llm_compatible_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_def.name,
                            "description": tool_def.description,
                            "parameters": tool_def.inputSchema
                        }
                    })
                else:
                    llm_compatible_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_def.name,
                            "description": tool_def.description,
                            "parameters": {"type": "object", "properties": {}}
                        }
                    })
            return llm_compatible_tools
        except Exception as e:
            print(f"ERROR: Could not discover tools from MCP server: {e}")
            return []

    async def _call_llm(self, messages: list, tools_for_llm: list = None) -> dict:
        """
        Internal method to send requests to the watsonx.ai LLM.
        Adjusted to use the watsonx.ai Python SDK.
        """
        if self.wx_api_key == "YOUR_WATSONX_API_KEY_HERE" or self.wx_project_id == "YOUR_WATSONX_PROJECT_ID_HERE":
            print("ERROR: watsonx.ai API key or Project ID not configured. Please set WX_API_KEY and WX_PROJECT_ID.")
            return {"error": "watsonx.ai not configured."}

        # watsonx.ai generally works with a single string prompt or a list of messages for chat models.
        # Tooling in watsonx.ai might require specific prompt formatting or use of specific models.
        # For simplicity, we'll try to adapt the OpenAI-like message structure to watsonx.ai.

        # Convert conversation history to a single string for non-chat models,
        # or use specific chat model methods if available and if the model supports it.
        # For tool calling, watsonx.ai often expects tool definitions to be injected into the prompt.
        
        # This is a simplified approach. For robust tool calling with watsonx.ai,
        # you might need a more sophisticated prompt engineering strategy or
        # leverage frameworks like LangChain's watsonx.ai integration which abstract this.

        # Let's create a combined prompt for watsonx.ai to handle both conversation and tools
        prompt_text = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt_text += f"Assistant: {msg['content']}\n"
            elif msg["role"] == "tool":
                # For tool results, the LLM needs to know what the tool returned
                prompt_text += f"Tool Output ({msg['name']}): {msg['content']}\n"

        if tools_for_llm:
            # Inject tool definitions into the prompt.
            # This is a common pattern for LLMs that don't have native "tools" parameters.
            # The exact format might need tuning based on the specific watsonx.ai model.
            prompt_text += "\nAvailable Tools:\n"
            for tool_def in tools_for_llm:
                func = tool_def["function"]
                prompt_text += f"- {func['name']}: {func['description']}"
                if func["parameters"] and func["parameters"].get("properties"):
                    props = ", ".join(func["parameters"]["properties"].keys())
                    prompt_text += f" (Parameters: {props})"
                prompt_text += "\n"
            prompt_text += "\nIf a tool is needed, respond with:\n"
            prompt_text += "```json\n{\n  \"tool_calls\": [\n    {\n      \"function\": {\n        \"name\": \"tool_name\",\n        \"arguments\": { \"arg1\": \"value1\" }\n      }\n    }\n  ]\n}\n```\n"
            prompt_text += "Otherwise, respond directly.\n"
        
        prompt_text += "Assistant:" # Indicate it's the assistant's turn to respond

        print(f"DEBUG: Calling watsonx.ai LLM with prompt:\n{prompt_text}")

        try:
            # watsonx.ai's generate_text method takes a single string prompt
            response_text = self.llm_model.generate_text(prompt=prompt_text)
            print(f"DEBUG: watsonx.ai response: {response_text}")

            # Now, parse the watsonx.ai response to check for tool calls or direct answers
            # This parsing needs to be robust as it relies on the LLM's output format.
            # We'll try to detect the JSON for tool calls.
            if "```json" in response_text and "```" in response_text:
                try:
                    json_start = response_text.find("```json") + len("```json")
                    json_end = response_text.find("```", json_start)
                    json_str = response_text[json_start:json_end].strip()
                    llm_output = json.loads(json_str)
                    
                    # Assume watsonx.ai will follow the OpenAI-like tool_calls structure
                    # if we prompt it correctly.
                    if "tool_calls" in llm_output:
                        return {"choices": [{"message": {"tool_calls": llm_output["tool_calls"]}}]}
                    else:
                        # If it was a JSON but not tool_calls, treat as direct content
                        return {"choices": [{"message": {"content": response_text}}]}
                except json.JSONDecodeError:
                    print("WARNING: LLM response contained malformed JSON.")
                    # Fallback to treating it as direct text if JSON parsing fails
                    return {"choices": [{"message": {"content": response_text}}]}
            else:
                # No JSON detected, treat as direct text response
                return {"choices": [{"message": {"content": response_text}}]}

        except Exception as e:
            print(f"Error calling watsonx.ai LLM: {e}")
            return {"error": str(e)}

    async def send_query(self, query: str) -> str:
        """
        Sends a query to the LLM and orchestrates tool use via the MCP server.
        """
        self.conversation_history.append({"role": "user", "content": query})

        if not hasattr(self, 'available_tools_schema'):
            self.available_tools_schema = await self._discover_tools()

        llm_response = await self._call_llm(self.conversation_history, self.available_tools_schema)

        if "error" in llm_response:
            return f"An error occurred with LLM: {llm_response['error']}"

        message = llm_response.get("choices", [{}])[0].get("message", {})
        
        if "tool_calls" in message:
            tool_calls = message["tool_calls"]
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args_str = json.dumps(tool_call["function"]["arguments"]) # Ensure it's a string for json.loads

                print(f"DEBUG: LLM wants to call tool: {tool_name} with args: {tool_args_str}")

                try:
                    tool_args = tool_call["function"]["arguments"] # Already parsed from LLM output
                    
                    async with self.mcp_client:
                        # mcp_tool_execution_result = await self.mcp_client.call_tool(tool_name, {"prompt": query})
                        mcp_tool_execution_result = await self.mcp_client.call_tool(tool_name, tool_args)
                    
                    if mcp_tool_execution_result.is_error:
                        error_content_parts = []
                        for content_block in mcp_tool_execution_result.content:
                            if hasattr(content_block, 'text'):
                                error_content_parts.append(content_block.text)
                            else:
                                error_content_parts.append(str(content_block))
                        return f"Error executing tool '{tool_name}' via MCP server: {''.join(error_content_parts) or 'Unknown error'}"
                    else:
                        tool_output = mcp_tool_execution_result.content 
                        print(f"DEBUG: Tool '{tool_name}' returned: {tool_output}")

                        return tool_output

                except Exception as e:
                    return f"Error orchestrating tool '{tool_name}': {e}"
        else:
            return message.get("content", "No direct answer from LLM.")

# --- Example Usage ---
async def main():
    # Load credentials from environment variables or use placeholders
    wx_api_key = os.getenv("WATSONX_API_KEY", WX_API_KEY)
    wx_project_id = os.getenv("WATSONX_PROJECT_ID", WX_PROJECT_ID)
    wx_api_url = os.getenv("WATSONX_API_URL", WX_API_URL)

    if wx_api_key == "YOUR_WATSONX_API_KEY_HERE" or wx_project_id == "YOUR_WATSONX_PROJECT_ID_HERE":
        print("\n--- CONFIGURATION REQUIRED ---")
        print("Please set your WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables,")
        print("or replace the placeholders in the script with your actual watsonx.ai credentials.")
        print("Without these, the program cannot interact with watsonx.ai and will print an error message instead.")
        print("----------------------------------------------------------------------\n")
        return

    client = MCPClient(MCP_SERVER_URL, wx_api_key, wx_project_id, wx_api_url, WX_MODEL_ID)
    
    async with client.mcp_client:
        # print(f"Attempting to connect to MCP Server: {client.mcp_client.base_url}")
        client.available_tools_schema = await client._discover_tools()
        if not client.available_tools_schema:
            print("ERROR: Failed to discover tools from MCP Server. Ensure it's running and accessible at the specified URL.")
            print("The client cannot proceed without discovering tools.")
            return

        print(f"Discovered tools: {[t['function']['name'] for t in client.available_tools_schema]}")
        print("\n--- Starting Query Session ---")
        print("-------------------------------------------------------------\n")

        query = "List the resource groups"
        print(f"\nUser: {query}")
        response1 = await client.send_query(query)
        print(f"Agent: {response1}")

if __name__ == "__main__":
    asyncio.run(main())
