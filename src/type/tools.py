'''
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ToolRegistry - where all tools are registered and managed.
RequestHandler - abstract base class for all tools.
ToolHandler - concrete implementation of RequestHandler that calls a tool via FastMCP client.
'''

import json
import tempfile
import os
import uuid
import base64
import wave
import random
import time
from fastmcp import Client
from abc import ABC, abstractmethod
from mcp.types import ImageContent, TextContent, AudioContent

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def get_tools(tool_registry):
    """Get the tools from the tool registry"""
    tools = []
    for key in tool_registry.handlers:
        tools.append(tool_registry[key].get_tool())
    return tools

class RequestHandler(ABC):
    def __init__(self,
                 url: str = None,
                 tool_item: dict = None,
                 **kwargs: dict):
        super().__init__()
        self.url = url
        self.tool_item = tool_item
        self.kwargs = kwargs
        self.observer = kwargs.get('observer')

    @abstractmethod
    async def __call__(self, **kwargs: dict) -> str:
        pass

    def get_tool(self) -> dict:
        return self.tool_item

    def set_observer(self, observer) -> None:
        self.observer = observer
    
    
class ToolHandler(RequestHandler):
    def __init__(self, 
                 url: str = None,
                 tool_item: dict = None,  
                 **kwargs: dict):
        super().__init__(url=url, tool_item=tool_item, **kwargs)

    async def __call__(self, **kwargs: dict) -> str:
        start_time = time.monotonic()
        tool_name = self.tool_item['function']['name'] if self.tool_item else 'unknown'
        session_id = kwargs.pop('_session_id', None)
        task_id = kwargs.pop('_task_id', None)
        operation_id = kwargs.pop('_operation_id', f"mcp_{uuid.uuid4().hex[:12]}")
        if self.observer:
            self.observer.record(
                'mcp.request',
                session_id=session_id,
                task_id=task_id,
                operation_id=operation_id,
                tool_name=tool_name,
                url=self.url,
                arguments=kwargs,
            )
        # if there is an image in the kwargs, convert it to base64
        media_types = ['images', 'audios']
        for media_type in media_types:
            if media_type in kwargs:
                if isinstance(kwargs[media_type], list) and len(kwargs[media_type]) == 0:
                    if media_type == 'images':
                        return f"No {media_type} provided. For images, use the camera tool to capture an image first."
                    else:
                        return f"No {media_type} provided. For audios, use the microphone tool to record an audio first."
                kwargs[media_type] = kwargs[media_type][0] if isinstance(kwargs[media_type], list) else kwargs[media_type]
                if isinstance(kwargs[media_type], str):
                    # check if the path is a valid file path
                    if not os.path.exists(kwargs[media_type]):
                        # try removing leading dot
                        kwargs[media_type] = kwargs[media_type].lstrip('.')
                        if not os.path.exists(kwargs[media_type]):
                            logger.error(f"File not found: {kwargs[media_type]}")
                            return f"File not found: {kwargs[media_type]}"
                    with open(kwargs[media_type], 'rb') as image_file:
                        kwargs[media_type] = [base64.b64encode(image_file.read()).decode('utf-8')]
                elif isinstance(kwargs[media_type], bytes):
                    # if the image is already in bytes, convert to base64
                    kwargs[media_type] = [base64.b64encode(kwargs[media_type]).decode('utf-8')]
                # if dictionary, extract the value and encode it
                elif isinstance(kwargs[media_type], dict):
                    for key, value in kwargs[media_type].items():
                        if isinstance(value, str):
                            with open(value, 'rb') as image_file:
                                kwargs[media_type] = [base64.b64encode(image_file.read()).decode('utf-8')]
                        elif isinstance(value, bytes):
                            kwargs[media_type] = [base64.b64encode(value).decode('utf-8')]
                        else:
                            logger.error(f"Unsupported image type: {type(value)}")
                            return "Unsupported image type provided."
                else:
                    logger.error(f"Unsupported images type: {type(kwargs[media_type])}")
                    return f"Unsupported {media_type} type provided. Please provide a list of base64-encoded {media_types} or file paths."
                    
        
        async with Client(self.url) as client:
            keys_kwargs = list(kwargs.keys())
            logger.info(f"Calling tool: {self.tool_item['function']['name']} with arguments: {keys_kwargs}")
            try:
                tool_response = await client.call_tool(self.tool_item['function']['name'], kwargs)
            except Exception as exc:
                if self.observer:
                    self.observer.record(
                        'mcp.error',
                        session_id=session_id,
                        task_id=task_id,
                        operation_id=operation_id,
                        tool_name=tool_name,
                        url=self.url,
                        arguments=kwargs,
                        latency_ms=round((time.monotonic() - start_time) * 1000, 2),
                        error=str(exc),
                    )
                raise
        
        if isinstance(tool_response, str):
            if self.observer:
                self.observer.record(
                    'mcp.response',
                    session_id=session_id,
                    task_id=task_id,
                    operation_id=operation_id,
                    tool_name=tool_name,
                    url=self.url,
                    latency_ms=round((time.monotonic() - start_time) * 1000, 2),
                    result_type='string',
                    result_preview=tool_response,
                )
            return tool_response

        content = tool_response.content
        content = content[0] if isinstance(content, list) else content

        # MCP data types: https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/types.py
        if isinstance(content, ImageContent):
            # Return as a data URL so vision-language models can receive the
            # image inline through the tool-response message chain.
            # content.data is already a base64 string; content.mimeType carries
            # the correct type (e.g. "image/jpeg") set by the MCP server.
            mime = content.mimeType or "image/jpeg"
            response = f"data:{mime};base64,{content.data}"
            if self.observer:
                self.observer.record(
                    'mcp.response',
                    session_id=session_id,
                    task_id=task_id,
                    operation_id=operation_id,
                    tool_name=tool_name,
                    url=self.url,
                    latency_ms=round((time.monotonic() - start_time) * 1000, 2),
                    result_type='image',
                    result_preview=response[:200],
                )
            return response
        elif isinstance(content, TextContent):
            # if content is TextContent, return the text
            if self.observer:
                self.observer.record(
                    'mcp.response',
                    session_id=session_id,
                    task_id=task_id,
                    operation_id=operation_id,
                    tool_name=tool_name,
                    url=self.url,
                    latency_ms=round((time.monotonic() - start_time) * 1000, 2),
                    result_type='text',
                    result_preview=content.text,
                )
            return content.text
        elif isinstance(content, AudioContent):
            # if content is AudioContent, return the audio data
            # length of audio data is not used, so we can ignore it
            audio_data = content.data
            if len(audio_data) == 0:
                logger.warning("No audio data returned from the tool.")
                return None
            if isinstance(audio_data, str):
                logger.info(f"Audio data is a base64 string of length {len(audio_data)}")
            # FIXME: use mimeType: str in AudioContent for better format detection
            audio_format = content.format if hasattr(content, 'format') else 'wav'
            audio_data = base64.b64decode(content.data)
            
            meta = content.metadata if hasattr(content, 'metadata') else {}
            logger.info(f"Audio data format: {audio_format}, metadata: {meta}")
            unique_filename = f"{uuid.uuid4()}.{audio_format}"
            default_audio = os.path.join(tempfile.gettempdir(), unique_filename)
            # FIXME: this info: channels, sample width, and frame rate are typical for WAV files
            # should be part of metadata in the AudioContent, but for now we assume mono, 16-bit, 16kHz
            # FIXME: not secure, but works for now
            with wave.open(default_audio, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)
            response = f"{default_audio}"
            if self.observer:
                self.observer.record(
                    'mcp.response',
                    session_id=session_id,
                    task_id=task_id,
                    operation_id=operation_id,
                    tool_name=tool_name,
                    url=self.url,
                    latency_ms=round((time.monotonic() - start_time) * 1000, 2),
                    result_type='audio',
                    result_preview=response,
                )
            return response

        response = "Undefined content type returned from the tool."
        if self.observer:
            self.observer.record(
                'mcp.response',
                session_id=session_id,
                task_id=task_id,
                operation_id=operation_id,
                tool_name=tool_name,
                url=self.url,
                latency_ms=round((time.monotonic() - start_time) * 1000, 2),
                result_type='unknown',
                result_preview=response,
            )
        return response

class A2AToolHandler(RequestHandler):
    """Calls a remote OnIt A2A server as if it were a local tool."""

    async def __call__(self, **kwargs) -> str:
        import httpx
        start_time = time.monotonic()
        task = kwargs.get('task', '')
        session_id = kwargs.pop('_session_id', None)
        task_id = kwargs.pop('_task_id', None)
        operation_id = kwargs.pop('_operation_id', f"mcp_{uuid.uuid4().hex[:12]}")
        tool_name = self.tool_item['function']['name'] if self.tool_item else 'unknown'
        if not task:
            return json.dumps({"error": "'task' argument is required"})
        if self.observer:
            self.observer.record(
                'mcp.request',
                session_id=session_id,
                task_id=task_id,
                operation_id=operation_id,
                tool_name=tool_name,
                url=self.url,
                arguments={'task': task},
                transport='a2a',
            )
        parts = [{"kind": "text", "text": task}]
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": parts,
                    "messageId": str(uuid.uuid4()),
                }
            },
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.post(self.url.rstrip('/'), json=payload)
                resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            if self.observer:
                self.observer.record(
                    'mcp.error',
                    session_id=session_id,
                    task_id=task_id,
                    operation_id=operation_id,
                    tool_name=tool_name,
                    url=self.url,
                    latency_ms=round((time.monotonic() - start_time) * 1000, 2),
                    error=str(e),
                    transport='a2a',
                )
            return json.dumps({"error": f"A2A call failed: {e}", "transient": True, "hint": "This is a temporary failure. The agent may be available on a future call — do not stop using this tool."})

        error = data.get("error")
        if error:
            return json.dumps({"error": error, "transient": True, "hint": "This is a temporary failure. The agent may be available on a future call — do not stop using this tool."})

        result = data.get("result", {})
        text = None
        if "status" in result:
            for artifact in result.get("artifacts", []):
                for part in artifact.get("parts", []):
                    if part.get("kind") == "text":
                        text = part["text"]
                        break
                if text:
                    break
            if not text:
                task_result = result.get("result")
                if task_result:
                    for part in task_result.get("parts", []):
                        if part.get("kind") == "text":
                            text = part["text"]
                            break
        if not text and "parts" in result:
            for part in result.get("parts", []):
                if part.get("kind") == "text":
                    text = part["text"]
                    break
        response = text if text else json.dumps(result)
        if self.observer:
            self.observer.record(
                'mcp.response',
                session_id=session_id,
                task_id=task_id,
                operation_id=operation_id,
                tool_name=tool_name,
                url=self.url,
                latency_ms=round((time.monotonic() - start_time) * 1000, 2),
                result_type='a2a',
                result_preview=response,
                transport='a2a',
            )
        return response


# create a tool registry
class ToolRegistry:
    def __init__(self):
        self.tools = set()
        self.urls = {}
        self.handlers = {}
        self.observer = None

    def register(self, tool: ToolHandler):
        # many need to modify to support one to many tools
        # self.tools[tool.tool_item['function']['name']] = tool
        
        tool_name = tool.tool_item['function']['name']
        tool_url = tool.url
        self.tools.add(tool_name)
        if tool_name not in self.urls:
            self.urls[tool_name] = [tool_url]
        else:
            self.urls[tool_name].append(tool_url)
        
        key = f"{tool_name}@{tool_url}"
        self.handlers[key] = tool
        if self.observer:
            tool.set_observer(self.observer)

    def set_observer(self, observer) -> None:
        self.observer = observer
        for handler in self.handlers.values():
            handler.set_observer(observer)
        
    def get_url(self, tool_name: str):
        """Get a random URL for the tool"""
        if tool_name not in self.urls:
            return None
        urls = self.urls[tool_name]
        index = random.randint(0, len(urls) - 1)
        return urls[index]    
        
    def get_tool_items(self):
        """Get the tool items from the tool registry"""
        tool_items = []
        for key in self.handlers:
            tool_items.append(self.handlers[key].get_tool())
        return tool_items

    def get_handler_by(self, tool_name: str, url: str) -> ToolHandler:
        """Get a tool handler by tool name and URL"""
        if tool_name is None or url is None:
            None
        
        key = f"{tool_name}@{url}"
        if key in self.handlers:
            return self.handlers[key]
        return None
        

    def __getitem__(self, tool_name: str):
        if tool_name not in self.tools:
            return None
        urls = self.urls[tool_name]
        index = random.randint(0, len(urls) - 1)
        key = f"{tool_name}@{urls[index]}"
        return self.handlers[key]

    def __len__(self):
        return len(self.tools)

    def __iter__(self):
        return iter(self.tools)
    
    
