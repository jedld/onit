'''

Tool related utility functions for On-it agent.

'''

from fastmcp import Client
from rich.status import Status
from typing import Tuple

import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from type.tools import *

# Configure logging
import logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _discover_server_tools(server: dict) -> list[ToolHandler]:
    """Discover tools from a single MCP server.

    Args:
        server: MCP server configuration dict.

    Returns:
        List of ToolHandler instances discovered from this server.
    """
    enabled = server.get('enabled', True)
    if not enabled:
        logger.info(f"[discover_tools] Skipping disabled MCP server: {server.get('name', 'Unknown')}")
        return []
    url = server.get('url', None)
    if not url:
        logger.error(f"[discover_tools] MCP server URL is not provided.")
        return []
    logger.info(f"[discover_tools] Discovering tools from MCP server: {url}")
    handlers = []
    async with Client(url) as client:
        tools_list = await client.list_tools()
        resources_list = await client.list_resources()
        tools_list.extend(resources_list)

        for tool_item in tools_list:
            if hasattr(tool_item, 'inputSchema'):
                parameters = {
                    'type': 'object',
                    'properties': tool_item.inputSchema['properties'],
                }
            else:
                parameters = {}
                if hasattr(tool_item, 'arguments') and tool_item.arguments:
                    properties = {}
                    for arg in tool_item.arguments:
                        prop = {'type': 'string'}
                        if arg.description:
                            prop['description'] = arg.description
                        properties[arg.name] = prop
                    parameters = {
                        'type': 'object',
                        'properties': properties,
                    }
                else:
                    for attr, value in tool_item.__dict__.items():
                        if hasattr(value, 'model_dump'):
                            parameters[attr] = value.model_dump()
                        elif isinstance(value, list):
                            parameters[attr] = [
                                v.model_dump() if hasattr(v, 'model_dump') else v
                                for v in value
                            ]
                        else:
                            parameters[attr] = value
            returns = None
            if hasattr(tool_item, 'outputSchema'):
                returns = tool_item.outputSchema
            else:
                returns = {}
            if returns is not None:
                returns = returns['properties'] if 'properties' in returns else {}

            tool_entry = {
                'type': 'function',
                'function': {
                    'name': tool_item.name,
                    'description': tool_item.description,
                    'parameters': parameters,
                    'returns': returns,
                    }
            }
            handler = ToolHandler(url=url, tool_item=tool_entry)
            handlers.append(handler)
            logger.info(f"[discover_tools] {tool_entry}")

        logger.info(f"[discover_tools] {tools_list}")
    return handlers


async def _discover_a2a_agent(agent: dict) -> list:
    """Discover a remote OnIt A2A agent and register it as a callable tool.

    Fetches the agent card from {url}/.well-known/agent.json and creates an
    A2AToolHandler with a single 'task' parameter.
    """
    if not agent.get('enabled', True):
        logger.info(f"[discover_tools] Skipping disabled A2A agent: {agent.get('name', 'Unknown')}")
        return []

    url = agent.get('url', '').rstrip('/')
    if not url:
        logger.error("[discover_tools] A2A agent has no URL.")
        return []

    # Derive tool name: prefer explicit 'tool_name', fall back to slugified 'name'
    tool_name = agent.get('tool_name') or agent.get('name', '')
    tool_name = re.sub(r'[^a-zA-Z0-9]+', '_', tool_name).strip('_').lower()
    if not tool_name:
        tool_name = 'ask_agent'

    # Try to fetch agent card for description
    description = agent.get('description', f'Delegate a task to the {tool_name} agent.')
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/.well-known/agent.json")
            if resp.status_code == 200:
                card = resp.json()
                description = card.get('description', description)
                if not tool_name or tool_name == 'ask_agent':
                    card_name = re.sub(r'[^a-zA-Z0-9]+', '_', card.get('name', '')).strip('_').lower()
                    if card_name:
                        tool_name = card_name
    except Exception as e:
        logger.warning(f"[discover_tools] Could not fetch agent card from {url}: {e}")

    tool_entry = {
        'type': 'function',
        'function': {
            'name': tool_name,
            'description': description,
            'parameters': {
                'type': 'object',
                'properties': {
                    'task': {
                        'type': 'string',
                        'description': 'The task or question to delegate to this agent.',
                    }
                },
                'required': ['task'],
            },
            'returns': {},
        }
    }
    handler = A2AToolHandler(url=url, tool_item=tool_entry)
    logger.info(f"[discover_tools] Registered A2A agent tool: {tool_name} -> {url}")
    return [handler]


async def discover_tools(mcp_servers: list, a2a_agents: list = None) -> Tuple[ToolRegistry]:
    """
    Automatically discover tools from all MCP servers and A2A agents in parallel.

    Args:
        mcp_servers (list): a list of MCP server configurations.
        a2a_agents (list): optional list of A2A agent configurations.

    Returns:
        Tuple[ToolRegistry]: a registry of all discovered tools.
    """
    import asyncio
    tool_registry = ToolRegistry()

    # Discover tools from all servers concurrently
    results = await asyncio.gather(
        *[_discover_server_tools(server) for server in mcp_servers],
        return_exceptions=True
    )

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            name = mcp_servers[i].get('name', 'Unknown')
            logger.error(f"[discover_tools] Failed to discover tools from {name}: {result}")
            continue
        for handler in result:
            tool_registry.register(handler)

    # Discover A2A agent tools
    if a2a_agents:
        a2a_results = await asyncio.gather(
            *[_discover_a2a_agent(agent) for agent in a2a_agents],
            return_exceptions=True
        )
        for i, result in enumerate(a2a_results):
            if isinstance(result, Exception):
                name = a2a_agents[i].get('name', 'Unknown')
                logger.error(f"[discover_tools] Failed to register A2A agent {name}: {result}")
                continue
            for handler in result:
                tool_registry.register(handler)

    logger.info(f"[discover_tools] Registered {len(tool_registry.tools)} tools")
    return tool_registry
    

async def listen(tool_registry: ToolRegistry, 
                 status: Status = None, 
                 prompt_color: str = "bold dark_blue",
                 is_safety_task : bool = False) -> str:
    microphone = tool_registry['microphone'] if 'microphone' in tool_registry.tools else None
    tts = tool_registry['speech_synthesis'] if 'speech_synthesis' in tool_registry.tools else None
    speaker = tool_registry['speaker'] if 'speaker' in tool_registry.tools else None
    asr = tool_registry['speech_recognition'] if 'speech_recognition' in tool_registry.tools else None
    if not microphone or not asr or not tts or not speaker:
        logger.error(f"[{listen.__name__}] Microphone, ASR, TTS or Speaker tool is not available.")
        return None
    # FIXME: Avoid using intermediate wav file
    wav_file = await microphone()
    if is_safety_task:
        return ""
    if wav_file is None:
        wav_file = await tts(text="Sorry! I did not hear you. Please say it again.")
        await speaker(audios=[wav_file])
        return None
    if status is not None:
        status.update(f"[{prompt_color}] 🤔 Hold on...[/]")
    logger.info(f"[{__name__}] wav file: {wav_file}")
    
    message = await asr(audios=[wav_file])
    return message
