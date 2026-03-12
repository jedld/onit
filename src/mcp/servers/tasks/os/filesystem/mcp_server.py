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

Document Search MCP Server

Search patterns in documents using command-line tools (grep, awk, sed, find, tr).
Supports plain text, PDF, markdown files with table understanding.
Provides context for LLM question answering and task completion.

Requires:
    pip install fastmcp pypdf pdfplumber

Core Tools:
1. search_document - Search for patterns in a document (text, PDF, markdown)
2. search_directory - Search for patterns across files in a directory
3. extract_tables - Extract tables from documents (PDF, markdown)
4. find_files - Find files matching patterns
5. transform_text - Transform text using sed/awk/tr operations
'''

import os
import subprocess
import tempfile
from typing import Annotated, Optional, Dict, Any
from pydantic import Field

from fastmcp import FastMCP

from src.mcp.servers.tasks.shared import (
    secure_makedirs as _secure_makedirs,
    search_document_impl,
    search_directory_impl,
    extract_tables_impl,
    find_files_impl,
    transform_text_impl,
    get_document_context_impl,
)

import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("Document Search MCP Server")

# Constants
DEFAULT_TIMEOUT = 60

# Data path for temporary file creation (set via options['data_path'] in run())
# All temp files are confined to this directory. Never use home folder.
DATA_PATH = os.path.join(tempfile.gettempdir(), "onit", "data")


def _validate_read_path(file_path: str) -> str:
    """Validate that the path is within DATA_PATH.
    Relative paths are resolved against DATA_PATH (not CWD).
    Returns resolved absolute path.
    Raises ValueError if outside allowed directory."""
    if not os.path.isabs(os.path.expanduser(file_path)):
        file_path = os.path.join(DATA_PATH, file_path)
    abs_path = os.path.realpath(os.path.expanduser(file_path))
    abs_data = os.path.realpath(os.path.expanduser(DATA_PATH))
    if abs_path.startswith(abs_data + os.sep) or abs_path == abs_data:
        return abs_path
    raise ValueError(
        f"Access denied. Path must be within: {abs_data}. Got: {abs_path}"
    )


def _validate_dir_path(dir_path: str) -> str:
    """Validate a directory path is within DATA_PATH.
    Relative paths are resolved against DATA_PATH (not CWD).
    Returns resolved absolute path.
    Raises ValueError if outside allowed directory."""
    if not os.path.isabs(os.path.expanduser(dir_path)):
        dir_path = os.path.join(DATA_PATH, dir_path)
    abs_path = os.path.realpath(os.path.expanduser(dir_path))
    abs_data = os.path.realpath(os.path.expanduser(DATA_PATH))
    if abs_path.startswith(abs_data + os.sep) or abs_path == abs_data:
        return abs_path
    raise ValueError(
        f"Directory access denied. Path must be within: {abs_data}. Got: {abs_path}"
    )


def _run_command(command: str, cwd: str = ".", timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Execute a shell command and return results."""
    try:
        work_dir = os.path.abspath(os.path.expanduser(cwd))
        if not os.path.isdir(work_dir):
            return {"error": f"Directory does not exist: {work_dir}", "status": "error"}
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=timeout,
            env={**os.environ, "TERM": "dumb"}
        )
        
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "status": "success" if result.returncode == 0 else "failed"
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout} seconds", "status": "timeout"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


# =============================================================================
# TOOL 1: SEARCH DOCUMENT
# =============================================================================

@mcp.tool(
    title="Search Document",
    description="""Search for a regex pattern in a single document file. Supports text, PDF, and markdown files.
Uses grep-like regex pattern matching and returns matching lines with surrounding context.

IMPORTANT - Required parameters:
- path: FULL absolute file path (e.g., "/tmp/onit/data/<session_id>/report.pdf"). Always use the complete working directory path from your system prompt — never use relative paths.
- pattern: Regex search pattern to find in the document (e.g., "error.*timeout", "subjects")
  Do NOT use 'query' - the parameter name is 'pattern'.

Optional parameters:
- case_sensitive: Whether search is case-sensitive (default: false)
- context_lines: Number of lines of context before/after each match (default: 3).
  Do NOT use 'context_chars' - the parameter name is 'context_lines'.
- max_matches: Maximum number of matches to return (default: 50).
  Do NOT use 'max_sections' - the parameter name is 'max_matches'.

Example: search_document(path="/tmp/onit/data/<session_id>/report.pdf", pattern="conclusion")

Returns JSON: {matches, total_matches, file, format, status}
Each match includes: {line_number, match, context_before, context_after}"""
)
def search_document(
    path: Annotated[Optional[str], Field(description="FULL absolute file path to search")] = None,
    pattern: Annotated[Optional[str], Field(description="Regex search pattern to find in the document (e.g., 'error.*timeout', 'subjects')")] = None,
    case_sensitive: Annotated[bool, Field(description="Whether search is case-sensitive")] = False,
    context_lines: Annotated[int, Field(description="Number of lines of context before/after each match")] = 3,
    max_matches: Annotated[int, Field(description="Maximum number of matches to return")] = 50
) -> str:
    return search_document_impl(
        path=path, pattern=pattern, case_sensitive=case_sensitive,
        context_lines=context_lines, max_matches=max_matches,
        validate_read_path=_validate_read_path,
    )


# =============================================================================
# TOOL 2: SEARCH DIRECTORY
# =============================================================================

@mcp.tool(
    title="Search Directory",
    description="""Search for patterns across files in a directory using grep.
Recursively searches text files matching the file pattern.

Args:
- directory: FULL absolute directory path (e.g., "/tmp/onit/data/<session_id>"). Always use the complete working directory path from your system prompt — never use relative paths.
- pattern: Search pattern (regex with -E flag)
- file_pattern: File glob pattern (default: "*" for all files)
- case_sensitive: Case-sensitive search (default: false)
- include_hidden: Include hidden files (default: false)
- max_results: Maximum results to return (default: 100)

Returns JSON: {results, total_files, total_matches, status}
Each result includes: {file, line_number, content}"""
)
def search_directory(
    directory: Optional[str] = None,
    pattern: Optional[str] = None,
    file_pattern: str = "*",
    case_sensitive: bool = False,
    include_hidden: bool = False,
    max_results: int = 100
) -> str:
    return search_directory_impl(
        directory=directory, pattern=pattern, file_pattern=file_pattern,
        case_sensitive=case_sensitive, include_hidden=include_hidden,
        max_results=max_results,
        validate_dir_path=_validate_dir_path,
        run_command=_run_command,
    )


# =============================================================================
# TOOL 3: EXTRACT TABLES
# =============================================================================

@mcp.tool(
    title="Extract Tables",
    description="""Extract tables from documents. Supports PDF and markdown files.
Tables are returned in a structured format with headers and rows.

Args:
- path: FULL absolute file path (e.g., "/tmp/onit/data/<session_id>/report.pdf"). Always use the complete working directory path from your system prompt — never use relative paths.
- table_index: Specific table index to extract (1-based, default: all)
- output_format: Output format - "json" or "markdown" (default: "json")

Returns JSON: {tables, total_tables, file, format, status}
Each table includes: {headers, rows, row_count, page (for PDF)}"""
)
def extract_tables(
    path: Optional[str] = None,
    table_index: Optional[int] = None,
    output_format: str = "json"
) -> str:
    return extract_tables_impl(
        path=path, table_index=table_index, output_format=output_format,
        validate_read_path=_validate_read_path,
    )


# =============================================================================
# TOOL 4: FIND FILES
# =============================================================================

@mcp.tool(
    title="Find Files",
    description="""Find files matching patterns using the find command.
Searches recursively from the specified directory.

Args:
- directory: FULL absolute directory path (e.g., "/tmp/onit/data/<session_id>"). Always use the complete working directory path from your system prompt — never use relative paths.
- name_pattern: File name pattern (glob, e.g., "*.py", "test_*")
- file_type: Type filter - "f" (file), "d" (directory), or None (all)
- max_depth: Maximum directory depth (default: unlimited)
- size_filter: Size filter (e.g., "+1M", "-100k", "50k")
- modified_days: Modified within N days (e.g., 7 for last week)
- max_results: Maximum results (default: 100)

Returns JSON: {files, total_files, directory, status}"""
)
def find_files(
    directory: str = ".",
    name_pattern: Optional[str] = None,
    file_type: Optional[str] = None,
    max_depth: Optional[int] = None,
    size_filter: Optional[str] = None,
    modified_days: Optional[int] = None,
    max_results: int = 100
) -> str:
    return find_files_impl(
        directory=directory, name_pattern=name_pattern, file_type=file_type,
        max_depth=max_depth, size_filter=size_filter, modified_days=modified_days,
        max_results=max_results,
        validate_dir_path=_validate_dir_path,
        run_command=_run_command,
    )


# =============================================================================
# TOOL 5: TRANSFORM TEXT
# =============================================================================

@mcp.tool(
    title="Transform Text",
    description="""Transform text using sed, awk, or tr commands.
Useful for extracting, replacing, or reformatting text content.

Args:
- input_text: Text to transform (or path to file if is_file=true)
- is_file: If true, input_text is treated as a file path (default: false)
- operation: Transformation type - "sed", "awk", or "tr"
- expression: The sed/awk/tr expression to apply
  - sed: e.g., "s/old/new/g", "/pattern/d"
  - awk: e.g., "{print $1}", "NR==1", "/pattern/{print}"
  - tr: e.g., "a-z A-Z" (translate), "-d '\\n'" (delete)

Returns JSON: {output, operation, expression, status}"""
)
def transform_text(
    input_text: Optional[str] = None,
    operation: Optional[str] = None,
    expression: Optional[str] = None,
    is_file: bool = False
) -> str:
    return transform_text_impl(
        input_text=input_text, operation=operation, expression=expression,
        is_file=is_file, data_path=DATA_PATH,
        validate_read_path=_validate_read_path,
        run_command=_run_command,
    )


# =============================================================================
# TOOL 6: GET DOCUMENT CONTEXT
# =============================================================================

@mcp.tool(
    title="Get Document Context",
    description="""Extract relevant context from a document for answering questions.
Searches for keywords and returns surrounding context that can support answers.

Args:
- path: FULL absolute file path (e.g., "/tmp/onit/data/<session_id>/document.pdf"). Always use the complete working directory path — never use relative paths.
- query: The question or topic to find context for
- keywords: Additional keywords to search (comma-separated)
- context_chars: Characters of context around matches (default: 500)
- max_sections: Maximum context sections to return (default: 5)

Returns JSON: {sections, query, file, status}
Each section includes: {content, relevance_keywords, position}"""
)
def get_document_context(
    path: Optional[str] = None,
    query: Optional[str] = None,
    keywords: Optional[str] = None,
    context_chars: int = 500,
    max_sections: int = 5
) -> str:
    return get_document_context_impl(
        path=path, query=query, keywords=keywords,
        context_chars=context_chars, max_sections=max_sections,
        validate_read_path=_validate_read_path,
    )


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18202,
    path: str = "/sse",
    options: dict = {}
) -> None:
    """Run the MCP server."""
    global DATA_PATH

    if 'verbose' in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    if 'data_path' in options:
        DATA_PATH = options['data_path']
    _secure_makedirs(os.path.abspath(os.path.expanduser(DATA_PATH)))

    logger.info(f"Starting Document Search MCP Server at {host}:{port}{path}")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info("Available tools: search_document, search_directory, extract_tables, find_files, transform_text, get_document_context")

    quiet = 'verbose' not in options
    if quiet:
        import uvicorn.config
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    mcp.run(transport=transport, host=host, port=port, path=path,
            uvicorn_config={"access_log": False, "log_level": "warning"} if quiet else {})

# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    run()
