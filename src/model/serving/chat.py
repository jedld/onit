"""
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

Chat function supporting private vLLM and OpenRouter.ai models via OpenAI-compatible API.
Provider is auto-detected from the host URL.
"""

import asyncio
import base64
import logging
import os
import json
import re
import tempfile
import uuid
from openai import AsyncOpenAI, OpenAIError, APITimeoutError
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


def _resolve_api_key(host: str, host_key: str = "EMPTY") -> str:
    """Resolve the API key based on the host URL.

    For OpenRouter hosts, use host_key param or OPENROUTER_API_KEY env var.
    For vLLM and other local hosts, default to "EMPTY".
    """
    if "openrouter.ai" in host:
        if host_key and host_key != "EMPTY":
            return host_key
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError(
                "OpenRouter requires an API key. Set it via:\n"
                "  - serving.host_key in the config YAML\n"
                "  - OPENROUTER_API_KEY environment variable"
            )
        return key
    return host_key


def _parse_tool_call_from_content(content: str, tool_registry) -> Optional[dict]:
    """Detect a raw JSON tool call in message content.

    Some models return tool calls as plain JSON in the response body instead of
    using the structured tool_calls field.  This function tries to parse the
    content and, if it looks like a valid tool call for a known tool, returns
    a dict with 'name' and 'arguments'.
    """
    if not content or not tool_registry:
        return None
    # Strip thinking tags if present
    text = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
    # Try to find a JSON object in the text
    start = text.find("{")
    if start == -1:
        return None
    # Find the matching closing brace, respecting JSON string literals
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        # JSON may be truncated (e.g. max_tokens cut it off).
        # Try regex fallback to extract tool name and arguments.
        return _parse_truncated_tool_call(text[start:], tool_registry)
    try:
        obj = json.loads(text[start:end])
    except json.JSONDecodeError:
        return _parse_truncated_tool_call(text[start:end], tool_registry)
    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
        if obj["name"] not in tool_registry.tools:
            return None
        return obj
    return None


def _strip_image_content(messages: list) -> int:
    """Remove all image_url content parts from messages in-place.

    Replaces multipart user messages that contain image_url items with a
    plain-text version, keeping only the text parts.  Tool messages that
    were already replaced with '[image captured — displayed below]' are
    left untouched.

    Returns the number of image items removed.
    """
    removed = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = [p for p in content if isinstance(p, dict) and p.get("type") == "text"]
            img_parts  = [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"]
            if img_parts:
                removed += len(img_parts)
                msg["content"] = " ".join(p["text"] for p in text_parts) if text_parts else ""
    return removed


def _save_data_url_images(data_urls: list[str], data_path: str | None) -> list[str]:
    """Decode data-URL images and save them to data_path (or tempdir).

    Returns a list of absolute file paths, one per data URL.
    The caller can embed these paths as markdown ``![alt](path)`` so the
    web UI's _extract_file_paths picks them up and serves them via /uploads/.
    """
    save_dir = data_path or tempfile.gettempdir()
    os.makedirs(save_dir, exist_ok=True)
    saved = []
    for url in data_urls:
        m = re.match(r'data:image/([^;]+);base64,(.+)', url, re.DOTALL)
        if not m:
            continue
        mime_sub = m.group(1).strip()
        b64 = m.group(2).replace("\n", "").replace("\r", "").strip()
        ext = "jpg" if mime_sub in ("jpeg", "jpg") else mime_sub
        fname = f"{uuid.uuid4()}.{ext}"
        fpath = os.path.join(save_dir, fname)
        try:
            with open(fpath, "wb") as f:
                f.write(base64.b64decode(b64))
            saved.append(fpath)
        except Exception:
            pass
    return saved


def _parse_truncated_tool_call(text: str, tool_registry) -> Optional[dict]:
    """Attempt to extract a tool call from truncated/malformed JSON.

    When the model's response is cut off (e.g. by max_tokens), the JSON may be
    incomplete.  This function uses regex to extract the tool name and any
    parseable arguments from the partial JSON.
    """
    # Extract the tool name
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if not name_match:
        return None
    tool_name = name_match.group(1)
    if tool_name not in tool_registry.tools:
        return None
    # Try to extract arguments object - find where "arguments" value starts
    args_match = re.search(r'"arguments"\s*:\s*\{', text)
    if not args_match:
        return {"name": tool_name, "arguments": {}}
    args_start = args_match.end() - 1  # include the opening brace
    # Try progressively larger substrings, closing any open braces
    # First try parsing as-is with closing braces appended
    args_text = text[args_start:]
    # Count unclosed braces (string-aware)
    depth = 0
    in_str = False
    esc = False
    last_valid = -1
    for i, ch in enumerate(args_text):
        if esc:
            esc = False
            continue
        if ch == '\\' and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                last_valid = i + 1
                break
    if last_valid > 0:
        try:
            args = json.loads(args_text[:last_valid])
            return {"name": tool_name, "arguments": args}
        except json.JSONDecodeError:
            pass
    # Arguments JSON is truncated - return with empty args so the tool can be
    # re-invoked by the model on the next iteration
    return {"name": tool_name, "arguments": {}}


def _looks_like_raw_tool_call(content: str) -> bool:
    """Check if content looks like a raw tool-call JSON that wasn't parsed.

    Returns True if the text contains patterns like {"name": "...", "arguments": ...}
    that indicate the model emitted a tool call as plain text.
    """
    if not content:
        return False
    text = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
    # Quick heuristic: must contain both "name" and "arguments" keys in JSON-like syntax
    return bool(re.search(r'"name"\s*:\s*"[^"]+"', text) and re.search(r'"arguments"\s*:', text))


def _extract_base64_file(tool_response: str, data_path: str) -> str:
    """Detect base64-encoded file data in a tool response and save it to disk.

    If the response is JSON containing a 'file_data_base64' field, decode it,
    write the file to data_path, and return a cleaned JSON string with the
    base64 data replaced by the local file path.  Otherwise return the
    original response unchanged.
    """
    try:
        data = json.loads(tool_response)
    except (json.JSONDecodeError, TypeError):
        return tool_response

    if not isinstance(data, dict) or "file_data_base64" not in data:
        return tool_response

    file_data_b64 = data.pop("file_data_base64")
    file_name = data.get("file_name", f"{uuid.uuid4()}.bin")
    safe_name = os.path.basename(file_name)
    filepath = os.path.join(data_path, safe_name)
    os.makedirs(data_path, exist_ok=True)

    file_bytes = base64.b64decode(file_data_b64)
    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(file_bytes)

    data["saved_path"] = filepath
    data["download_url"] = f"/uploads/{safe_name}"
    data["file_size_bytes"] = len(file_bytes)
    return json.dumps(data)


async def chat(host: str = "http://127.0.0.1:8001/v1",
         host_key: str = "EMPTY",
         model: str = "Qwen/Qwen3-8B",
         instruction: str = "Tell me more about yourself.",
         images: List[str]|str = None,
         tool_registry: Optional[Any] = None,
         timeout: int = None,
         stream: bool = False,
         think: bool = False,
         safety_queue: Optional[asyncio.Queue] = None,
         error_container: Optional[list] = None,
         **kwargs) -> Optional[str]:

    tools = tool_registry.get_tool_items() if tool_registry else []
    chat_ui = kwargs['chat_ui'] if 'chat_ui' in kwargs else None
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    data_path = kwargs.get('data_path', '')
    max_tokens = kwargs.get('max_tokens', 8192)
    memories = kwargs.get('memories', None)
    prompt_intro = kwargs.get('prompt_intro', "I am a helpful AI assistant. My name is OnIt.")

    images_bytes = []
    if isinstance(images, list):
        for image_path in images:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as image_file:
                    images_bytes.append(base64.b64encode(image_file.read()).decode('utf-8'))
            else:
                if chat_ui:
                    chat_ui.add_log(f"Image file {image_path} not found, proceeding without this image.", level="warning")
                elif verbose:
                    print(f"Image file {image_path} not found, proceeding without this image.")
    elif isinstance(images, str):
        image_path = images
        if os.path.exists(image_path):
            with open(image_path, 'rb') as image_file:
                images_bytes = [base64.b64encode(image_file.read()).decode('utf-8')]
        else:
            if chat_ui:
                chat_ui.add_log(f"Image file {image_path} not found, proceeding without this image.", level="warning")
            elif verbose:
                print(f"Image file {image_path} not found, proceeding without this image.")

    if images_bytes:
        messages = [{
            "role": "system",
            "content": (
                f"{prompt_intro} "
                "You are an expert vision-language assistant. Your task is to analyze images with high precision, "
                "reasoning step-by-step about visual elements and their spatial relationships (e.g., coordinates, "
                "relative positions like left/right/center). Always verify visual evidence before concluding. "
                "If a task requires external data, calculation, or specific actions beyond visual description, "
                "use the provided tools. Be concise, objective, and format your tool calls strictly according to schema."
            )
        }]
    else:
        messages = [{"role": "system", "content": prompt_intro}]

    # Inject session history BEFORE the current instruction so the model
    # sees prior context first and treats the latest user message as the
    # one to respond to.
    session_history = kwargs.get('session_history', None)
    if session_history:
        for entry in session_history:
            messages.append({"role": "user", "content": entry["task"]})
            messages.append({"role": "assistant", "content": entry["response"]})

    # Current instruction goes last so the model responds to it
    if images_bytes:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images_bytes[0]}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": instruction})   
        
    if not memories and not session_history:
        message = {'role': 'tool', 'content': '', 'name': '', 'parameters': {}, "tool_call_id": ''}
        messages.append(message)

    api_key = _resolve_api_key(host, host_key)
    client = AsyncOpenAI(base_url=host, api_key=api_key)

    if chat_ui:
        chat_ui.add_log(f"Starting chat with model: {model}", level="info")

    MAX_CHAT_ITERATIONS = 100
    MAX_REPEATED_TOOL_CALLS = 30
    iteration_count = 0
    tool_call_history = []  # list of (name, args_json) tuples
    non_vlm_mode = False  # set True after a "not a multimodal model" error

    while True:
        iteration_count += 1
        if iteration_count > MAX_CHAT_ITERATIONS:
            msg = f"I am sorry 😊. Could you try to rephrase or provide additional details?"
            if chat_ui:
                chat_ui.add_log(f"Chat loop exceeded {MAX_CHAT_ITERATIONS} iterations, stopping.", level="warning")
            elif verbose:
                print(f"Chat loop exceeded {MAX_CHAT_ITERATIONS} iterations, stopping.")
            return msg

        try:
            if not safety_queue.empty():
                logger.warning("Safety queue triggered before API call, exiting chat loop.")
                return None

            completion_kwargs = dict(
                model=model,
                messages=messages,
                stream=stream,
                tool_choice="auto",          # never "required"
                temperature=0.6,             # official recommendation
                top_p=0.95,
                max_tokens=max_tokens,             # cap to prevent runaway generation
                extra_body={
                    "top_k": 20,             # vLLM extension, important for Qwen3
                    "repetition_penalty": 1.05,  # helps break repetition loops
                    "chat_template_kwargs": {"enable_thinking": think},  # if not using CoT
                },
            )
            if tools: # and not images_bytes:  # vLLM doesn't support tools + images in the same message, so only include tools if no images are present
                completion_kwargs["tools"] = tools

            # In non-VLM mode strip any image_url content before every request
            if non_vlm_mode:
                _strip_image_content(messages)

            chat_completion = await client.chat.completions.create(**completion_kwargs)

            await asyncio.sleep(0.1)
            if not safety_queue.empty():
                logger.warning("Safety queue triggered after API call, exiting chat loop.")
                return None
        except APITimeoutError as e:
            error_message = f"Request to {host} timed out after {timeout} seconds."
            logger.error(error_message)
            if chat_ui:
                chat_ui.add_log(error_message, level="error")
            elif verbose:
                print(error_message)
            if error_container is not None:
                error_container.append(error_message)
            return None
        except OpenAIError as e:
            error_str = str(e)
            # Auto-recover: context length exceeded — reduce max_tokens to fit
            if "400" in error_str and "context length" in error_str and "maximum input length" in error_str:
                import re as _re
                m = _re.search(r"context length is only (\d+).*?maximum input length of (\d+)", error_str)
                if m:
                    ctx_len = int(m.group(1))
                    max_input = int(m.group(2))
                    new_max = max(256, max_input - 256)
                    warn = f"Context length exceeded ({ctx_len} total, {max_input} input). Reducing max_tokens from {max_tokens} to {new_max} and retrying."
                    max_tokens = new_max
                    if chat_ui:
                        chat_ui.add_log(warn, level="warning")
                    elif verbose:
                        print(warn)
                    continue
            # Auto-recover: model rejected image content because it is text-only
            if "not a multimodal model" in error_str or ("400" in error_str and "multimodal" in error_str):
                non_vlm_mode = True
                removed = _strip_image_content(messages)
                # Detect available vision tools for the hint
                vision_tools = []
                if tool_registry:
                    for tname in ("ask_vision_agent", "ask_cosmos_agent", "ask_vlm"):
                        if tname in tool_registry.tools:
                            vision_tools.append(tname)
                if vision_tools:
                    hint = (
                        f"Note: this model cannot process images directly. "
                        f"Use the {' or '.join(vision_tools)} tool to analyze any camera images instead."
                    )
                else:
                    hint = (
                        "Note: this model cannot process images directly. "
                        "Describe what the camera tool would normally show based on context, "
                        "or inform the user that a vision-capable model is required."
                    )
                warn = f"Model is not multimodal — stripped {removed} image(s) and retrying with vision tool hint."
                if chat_ui:
                    chat_ui.add_log(warn, level="warning")
                elif verbose:
                    print(warn)
                # Inject the hint into the last user message, or append a new one
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                        msg["content"] = msg["content"].rstrip() + f"\n\n[System hint: {hint}]"
                        break
                else:
                    messages.append({"role": "user", "content": f"[System hint: {hint}]"})
                continue  # retry without images
            error_message = f"Error communicating with {host}: {e}."
            logger.error(error_message)
            if chat_ui:
                chat_ui.add_log(error_message, level="warning")
            elif verbose:
                print(error_message)
            if error_container is not None:
                error_container.append(error_message)
            return None
        except Exception as e:
            error_message = f"Unexpected error: {e}"
            logger.error(error_message)
            if chat_ui:
                chat_ui.add_log(error_message, level="error")
            elif verbose:
                print(error_message)
            if error_container is not None:
                error_container.append(error_message)
            return error_message
            
        tool_calls = chat_completion.choices[0].message.tool_calls
        if tool_calls is None or len(tool_calls) == 0:
            last_response = chat_completion.choices[0].message.content
            # Check if the model returned a tool call as raw JSON in the content
            raw_tool = _parse_tool_call_from_content(last_response, tool_registry)
            if raw_tool:
                function_name = raw_tool["name"]
                function_arguments = raw_tool["arguments"]
                synthetic_id = f"call_{uuid.uuid4().hex[:24]}"
                if chat_ui:
                    chat_ui.add_log(f"Calling: {function_name}({function_arguments})", level="info")
                    chat_ui.render()
                elif verbose:
                    print(f"{function_name}({function_arguments})")
                messages.append({"role": "assistant", "content": last_response})
                for tool_name in tool_registry.tools:
                    if tool_name == function_name:
                        try:
                            tool_handler = tool_registry[tool_name]
                            try:
                                tool_response = await asyncio.wait_for(
                                    tool_handler(**function_arguments),
                                    timeout=timeout,
                                )
                            except asyncio.TimeoutError:
                                tool_response = f"- tool call timed out after {timeout} seconds. Tool might have succeeded but no response was received. Check expected output."
                                if chat_ui:
                                    chat_ui.add_log(f"{function_name} timed out after {timeout}s", level="warning")
                                elif verbose:
                                    print(f"{function_name} timed out after {timeout}s")
                            tool_response = "" if tool_response is None else str(tool_response)
                            if data_path and "file_data_base64" in tool_response:
                                tool_response = _extract_base64_file(tool_response, data_path)
                            tool_message = {'role': 'tool', 'content': tool_response, 'name': function_name, 'parameters': function_arguments, "tool_call_id": synthetic_id}
                            messages.append(tool_message)
                            truncated = tool_response[:500] + "..." if len(tool_response) > 500 else tool_response
                            if chat_ui:
                                chat_ui.add_log(f"{function_name}({function_arguments}) returned: {truncated}", level="debug")
                            elif verbose:
                                print(f"{function_name}({function_arguments}) returned: {truncated}")
                        except Exception as e:
                            if chat_ui:
                                chat_ui.add_log(f"{function_name}({function_arguments}) error: {e}", level="error")
                            elif verbose:
                                print(f"{function_name}({function_arguments}) encountered an error: {e}")
                            tool_message = {'role': 'tool', 'content': f'Error: {e}', 'name': function_name, 'parameters': function_arguments, "tool_call_id": synthetic_id}
                            messages.append(tool_message)
                        break
                else:
                    # Tool not found in registry
                    tool_message = {'role': 'tool', 'content': f'Error: tool {function_name} not found', 'name': function_name, 'parameters': function_arguments, "tool_call_id": synthetic_id}
                    messages.append(tool_message)
                # Check for repeated tool calls
                call_key = (function_name, json.dumps(function_arguments, sort_keys=True))
                tool_call_history.append(call_key)
                if tool_call_history.count(call_key) >= MAX_REPEATED_TOOL_CALLS:
                    msg = f"I am sorry 😊. Could you try to rephrase or provide additional details?"
                    if chat_ui:
                        chat_ui.add_log(f"Repeated tool call detected: {function_name} called {tool_call_history.count(call_key)} times with same args", level="warning")
                    elif verbose:
                        print(f"Repeated tool call detected: {function_name} called {tool_call_history.count(call_key)} times with same args")
                    return msg
                continue  # loop back for the model to generate the final response

            # Guard against returning raw tool-call JSON to the user.
            # If the content looks like a tool call but couldn't be parsed,
            # ask the model to retry without tools.
            if _looks_like_raw_tool_call(last_response):
                if chat_ui:
                    chat_ui.add_log("Model returned unparseable raw tool-call JSON, retrying without tools.", level="warning")
                elif verbose:
                    print("Model returned unparseable raw tool-call JSON, retrying without tools.")
                messages.append({"role": "assistant", "content": last_response})
                messages.append({"role": "user", "content": "Please provide your answer as plain text, not as a JSON tool call."})
                continue

            if "</think>" in last_response:
                last_response = last_response.split("</think>")[1]
            return last_response

        messages.append(chat_completion.choices[0].message)
        for tool in tool_calls:
            await asyncio.sleep(0.1)
            if not safety_queue.empty():
                if verbose:
                    print("Safety queue triggered, exiting chat loop.")
                return None
            function_name = tool.function.name
            function_arguments = json.loads(tool.function.arguments)
            if chat_ui:
                chat_ui.add_log(f"Calling: {function_name}({function_arguments})", level="info")
                chat_ui.render()
            elif verbose:
                print(f"{function_name}({function_arguments})")
            # Ensure the function is available, and then call it
            # FIXME: Possible that 2 or more tools have the same name?
            for tool_name in tool_registry.tools:
                if tool_name == function_name:
                    try:
                        tool_handler = tool_registry[tool_name]
                        try:
                            tool_response = await asyncio.wait_for(
                                tool_handler(**function_arguments),
                                timeout=timeout,
                            )
                        except asyncio.TimeoutError:
                            tool_response = f"- tool call timed out after {timeout} seconds. Tool might have succeeded but no response was received. Check expected output."
                            if chat_ui:
                                chat_ui.add_log(f"{function_name} timed out after {timeout}s", level="warning")
                        tool_response = "" if tool_response is None else str(tool_response)
                        # Extract base64 file data from tool response and save to disk
                        if data_path and "file_data_base64" in tool_response:
                            tool_response = _extract_base64_file(tool_response, data_path)
                        tool_message = {'role': 'tool', 'content': tool_response, 'name': tool.function.name, 'parameters': function_arguments, "tool_call_id": tool.id,}
                        messages.append(tool_message)

                        # Log tool response (truncated for display)
                        truncated_response = tool_response[:200] + "..." if len(tool_response) > 200 else tool_response
                        if chat_ui:
                            chat_ui.add_log(f"{function_name}({function_arguments}) returned: {truncated_response}", level="debug")
                    except Exception as e:
                        if chat_ui:
                            chat_ui.add_log(f"{tool_name} error: {e}", level="error")
                        elif verbose:
                            print(f"{tool_name} encountered an error: {e}")
                        tool_message = {'role': 'tool', 'content': f'Error: {e}', 'name': tool.function.name, 'parameters': function_arguments, "tool_call_id": tool.id}
                        messages.append(tool_message)
                    break
            else:
                # Tool not found in registry
                tool_message = {'role': 'tool', 'content': f'Error: tool {function_name} not found', 'name': function_name, 'parameters': function_arguments, "tool_call_id": tool.id}
                messages.append(tool_message)
            # Check for repeated tool calls
            call_key = (function_name, json.dumps(function_arguments, sort_keys=True))
            tool_call_history.append(call_key)
            if tool_call_history.count(call_key) >= MAX_REPEATED_TOOL_CALLS:
                msg = f"I am sorry 😊. Could you try to rephrase or provide additional details?"
                if chat_ui:
                    chat_ui.add_log(f"Repeated tool call detected: {function_name} called {tool_call_history.count(call_key)} times with same args", level="warning")
                elif verbose:
                    print(f"Repeated tool call detected: {function_name} called {tool_call_history.count(call_key)} times with same args")
                return msg
