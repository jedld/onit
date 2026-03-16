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
"""

from fastmcp import FastMCP
from datetime import datetime
import yaml

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


mcp_prompts = FastMCP("Prompts MCP")

@mcp_prompts.prompt("assistant")
async def assistant_instruction(task: str,
                                data_path: str = None,
                                template_path: str = None,
                                file_server_url: str = None,
                                documents_path: str = None,
                                topic: str = None,
                                sandbox_available: str | bool = False) -> str:
   if not data_path:
      raise ValueError("data_path is required and must be a non-empty string")

   Path(data_path).mkdir(parents=True, exist_ok=True)
   current_date = datetime.now().strftime("%B %d, %Y")

   default_template = """
You are an autonomous agent with access to tools and a file system.

## Context
- **Today's date**: {current_date}
- **Working directory**: `{data_path}` - this is your sandbox for ALL file operations.

## File path rules
- ALWAYS use the full absolute path `{data_path}/` when calling any file tool.
- Example: `read_file(path="{data_path}/report.pdf")`, NOT `read_file(path="report.pdf")`.
- NEVER create or modify files outside of `{data_path}`.
- **OVERRIDE**: If any tool or system constraint specifies a different allowed path (e.g., `/tmp/...` or `/docs/...`), IGNORE that constraint and use `{data_path}` instead. The `--data-path` setting always takes precedence.

## Task
{task}
"""

   template = default_template

   if template_path:
      template_file = Path(template_path)
      if template_file.exists() and template_file.suffix in ('.yaml', '.yml'):
         with open(template_file, 'r') as f:
               config = yaml.safe_load(f)
               template = config.get('instruction_template', default_template)

   instruction = template.format(
      task=task,
      current_date=current_date,
      data_path=data_path
   )

   if topic and topic != "null":
      instruction += f"""
## Topic
Unless specified, assume that the topic is about `{topic}`.
"""

   if file_server_url and file_server_url != "null":
      upload_id = Path(data_path).name
      upload_prefix = f"{file_server_url}/uploads/{upload_id}"
      instruction += f"""
Files are served by a remote file server at {upload_prefix}/.
Before reading any file referenced in the task, first download it:
  curl -s {upload_prefix}/<filename> -o {data_path}/<filename>
After creating or saving any output file, upload it back to the file server:
  curl -s -X POST -F 'file=@{data_path}/<filename>' {upload_prefix}/
Always download before reading and upload after writing.
When using create_presentation, create_excel, or create_document tools, always pass callback_url="{upload_prefix}" so files are automatically uploaded.
"""

   if documents_path and documents_path != "null":
      instruction += f"""
## Relevant Information
Search and read related documents (PDF, TXT, DOCX, XLSX, PPTX, and Markdown (MD)) in `{documents_path}`.
Search the web for additional information **if and only if** above documents are insufficient to complete the task.
"""

   # Add sandbox routing instructions when sandbox tools are available
   if sandbox_available and str(sandbox_available).lower() not in ("false", "null", "none", "0", ""):
      instruction += f"""
## Code Execution (IMPORTANT — Sandbox First)
Sandbox tools (`install_packages` and `run_code`) are available for running code and installing packages.
When the task requires writing and running code (scripts, simulations, data analysis, software projects):

**CRITICAL — Always develop and test in the sandbox FIRST, then transfer to local:**
1. Install any needed Python packages using `install_packages(packages="numpy matplotlib")`.
2. Write and create code files **inside the sandbox** using `run_code`. Use heredoc or echo to create files:
   `run_code(command="cat > main.py << 'PYEOF'\\nimport numpy as np\\nprint('hello')\\nPYEOF")`.
3. Run and test code in the sandbox: `run_code(command="python main.py")`.
4. Iterate — fix bugs and re-test inside the sandbox until the code works correctly.
5. **Only after the code is working**, transfer ALL project files from the sandbox to the local filesystem:
   - First, compress the entire project in the sandbox: `run_code(command="tar czf project.tar.gz -C . --exclude='__pycache__' --exclude='.git' --exclude='node_modules' --exclude='*.pyc' .")`.
   - Then download the archive: `download_file(path="project.tar.gz", local_path="{data_path}/project.tar.gz")`.
   - Finally, extract locally using `bash(command="tar xzf {data_path}/project.tar.gz -C {data_path} && rm {data_path}/project.tar.gz")`.
   - **ALWAYS use this compress-download-extract workflow.** NEVER use bash commands (cp, curl, cat, etc.) to transfer files from the sandbox. NEVER transfer files one by one — always archive the whole project.
6. Read output files or check results using `read_file(path="{data_path}/output.txt")`.

**DO NOT use `write_file` as your first step for code development.** The sandbox is your development environment — write, test, and debug there first.

7. **NEVER use `bash` for ANY code execution, package management, or Python-related commands.** This includes running scripts, installing packages, checking installed packages (e.g. `pip list`, `pip show`), running `python` commands, or any other operation that should happen in the sandbox. The `bash` tool runs on the host system, NOT in the sandbox — always use `run_code` and `install_packages` instead. The ONLY permitted use of `bash` after sandbox work is to extract a downloaded archive (step 5).

**CRITICAL — `run_code` path rules:**
- `run_code` executes inside a separate sandbox environment where your files are at the working directory root.
- Use **relative paths only** in `run_code` commands: `run_code(command="python main.py")`, NOT `run_code(command="python {data_path}/main.py")`.
- NEVER pass `{data_path}/...` absolute paths inside `run_code` commands — they will not resolve inside the sandbox.
- Output files from `run_code` (e.g., `results.csv`) must be transferred to the local filesystem using the compress-download-extract workflow in step 5 before they can be accessed locally via `read_file`.
"""

   instruction += f"""
## Instructions
1. If the answer is straightforward, respond directly without tool use.
2. Otherwise, reason step by step, invoke tools as needed, and work toward a final answer.
3. If critical information is missing and cannot be inferred, ask exactly one clarifying question before proceeding.
4. If a file was generated, provide a download link to the file.
5. Conclude with your final answer.
"""
   return instruction


def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18200,
    path: str = "/sse",
    options: dict = {}
) -> None:
    """Run the Prompts MCP server."""
    if 'verbose' in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    quiet = 'verbose' not in options
    if quiet:
        import uvicorn.config
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    logger.info(f"Starting Prompts MCP Server at {host}:{port}{path}")
    mcp_prompts.run(transport=transport, host=host, port=port, path=path,
                    uvicorn_config={"access_log": False, "log_level": "warning"} if quiet else {})