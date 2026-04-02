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

Shared utilities for MCP task servers.

Common helper functions and tool logic shared between the bash and filesystem
MCP servers. Server-specific behavior (path validation, command execution) is
injected via callable parameters so each server retains its own security model.
'''

import json
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import logging

logger = logging.getLogger(__name__)

# Default constants (servers may override via their own module-level values)
MAX_OUTPUT_SIZE = 100000  # 100KB max output


# =============================================================================
# SHARED HELPER FUNCTIONS
# =============================================================================


def truncate_output(text: str, max_size: int = MAX_OUTPUT_SIZE) -> str:
    """Truncate output if it exceeds max size."""
    if len(text) > max_size:
        return text[:max_size] + f"\n\n... [OUTPUT TRUNCATED - {len(text)} bytes total]"
    return text


def secure_makedirs(dir_path: str) -> None:
    """Create directory with owner-only permissions (0o700)."""
    os.makedirs(dir_path, mode=0o700, exist_ok=True)


def validate_required(**kwargs) -> str:
    """Check for missing required arguments. Returns JSON error string or empty string."""
    missing = [name for name, value in kwargs.items() if value is None]
    if missing:
        return json.dumps({
            "error": f"Missing required argument(s): {', '.join(missing)}.",
            "status": "error"
        })
    return ""


def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("pypdf not installed")
        return ""
    except Exception as e:
        logger.error(f"Failed to read PDF: {e}")
        return ""


def extract_pdf_tables(file_path: str) -> List[Dict[str, Any]]:
    """Extract tables from PDF using pdfplumber."""
    tables = []
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                for table_idx, table in enumerate(page_tables, 1):
                    if table and len(table) > 0:
                        headers = table[0] if table else []
                        rows = table[1:] if len(table) > 1 else []
                        tables.append({
                            "page": page_num,
                            "table_index": table_idx,
                            "headers": headers,
                            "rows": rows,
                            "row_count": len(rows)
                        })
        return tables
    except ImportError:
        logger.warning("pdfplumber not installed. Run: pip install pdfplumber")
        return []
    except Exception as e:
        logger.error(f"Failed to extract tables from PDF: {e}")
        return []


def extract_markdown_tables(content: str) -> List[Dict[str, Any]]:
    """Extract tables from markdown content."""
    tables = []
    table_pattern = r'(\|[^\n]+\|\n\|[-:\| ]+\|\n(?:\|[^\n]+\|\n)*)'

    matches = re.finditer(table_pattern, content)
    for idx, match in enumerate(matches, 1):
        table_text = match.group(1)
        lines = table_text.strip().split('\n')

        if len(lines) >= 2:
            headers = [cell.strip() for cell in lines[0].split('|')[1:-1]]
            rows = []
            for line in lines[2:]:
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if cells:
                    rows.append(cells)

            tables.append({
                "table_index": idx,
                "headers": headers,
                "rows": rows,
                "row_count": len(rows),
                "raw": table_text
            })

    return tables


def get_file_content(file_path: str) -> tuple[str, str]:
    """Get file content and format. Returns (content, format)."""
    file_path = os.path.abspath(os.path.expanduser(file_path))

    if not os.path.isfile(file_path):
        return "", "error"

    ext = Path(file_path).suffix.lower()

    if ext == '.pdf':
        return extract_pdf_text(file_path), "pdf"
    else:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            if ext == '.md':
                return content, "markdown"
            else:
                return content, "text"
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return "", "error"


# =============================================================================
# SHARED TOOL LOGIC
# =============================================================================

# Each function below implements the core logic of an MCP tool. The caller
# (bash or filesystem server) passes in its own _validate_read_path,
# _validate_dir_path, _run_command, etc. so security behaviour stays
# server-specific.


def search_document_impl(
    path: Optional[str],
    pattern: Optional[str],
    case_sensitive: bool,
    context_lines: int,
    max_matches: int,
    validate_read_path: Callable[[str], str],
) -> str:
    """Core logic for search_document tool."""
    if err := validate_required(path=path, pattern=pattern):
        return err
    try:
        file_path = validate_read_path(path)

        if not os.path.isfile(file_path):
            return json.dumps({
                "error": f"File not found: {file_path}",
                "path": path,
                "status": "error"
            })

        content, file_format = get_file_content(file_path)

        if file_format == "error":
            return json.dumps({
                "error": "Failed to read file",
                "path": path,
                "status": "error"
            })

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return json.dumps({
                "error": f"Invalid regex pattern: {e}",
                "pattern": pattern,
                "status": "error"
            })

        lines = content.split('\n')
        matches = []

        for i, line in enumerate(lines):
            if regex.search(line):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)

                matches.append({
                    "line_number": i + 1,
                    "match": line.strip(),
                    "context_before": [l.strip() for l in lines[start:i]],
                    "context_after": [l.strip() for l in lines[i+1:end]]
                })

                if len(matches) >= max_matches:
                    break

        return json.dumps({
            "matches": matches,
            "total_matches": len(matches),
            "pattern": pattern,
            "file": file_path,
            "format": file_format,
            "truncated": len(matches) >= max_matches,
            "status": "success"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "path": path,
            "status": "error"
        })


def search_directory_impl(
    directory: Optional[str],
    pattern: Optional[str],
    file_pattern: str,
    case_sensitive: bool,
    include_hidden: bool,
    max_results: int,
    validate_dir_path: Callable[[str], str],
    run_command: Callable[..., Dict[str, Any]],
) -> str:
    """Core logic for search_directory tool."""
    if err := validate_required(directory=directory, pattern=pattern):
        return err
    try:
        dir_path = validate_dir_path(directory)

        if not os.path.isdir(dir_path):
            return json.dumps({
                "error": f"Directory not found: {dir_path}",
                "directory": directory,
                "status": "error"
            })

        grep_flags = "-rn"
        if not case_sensitive:
            grep_flags += "i"
        grep_flags += "E"

        exclude = "" if include_hidden else "--exclude-dir='.[!.]*' --exclude='.[!.]*'"

        cmd = f"grep {grep_flags} {exclude} --include={shlex.quote(file_pattern)} {shlex.quote(pattern)} . 2>/dev/null | head -n {int(max_results)}"

        result = run_command(cmd, cwd=dir_path)

        if result.get("status") == "error":
            return json.dumps(result)

        results = []
        output = result.get("stdout", "")

        if output:
            for line in output.split('\n'):
                if ':' in line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        results.append({
                            "file": parts[0],
                            "line_number": int(parts[1]) if parts[1].isdigit() else parts[1],
                            "content": parts[2].strip()
                        })

        return json.dumps({
            "results": results,
            "total_matches": len(results),
            "pattern": pattern,
            "directory": dir_path,
            "file_pattern": file_pattern,
            "truncated": len(results) >= max_results,
            "status": "success"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "directory": directory,
            "status": "error"
        })


def extract_tables_impl(
    path: Optional[str],
    table_index: Optional[int],
    output_format: str,
    validate_read_path: Callable[[str], str],
) -> str:
    """Core logic for extract_tables tool."""
    if err := validate_required(path=path):
        return err
    try:
        file_path = validate_read_path(path)

        if not os.path.isfile(file_path):
            return json.dumps({
                "error": f"File not found: {file_path}",
                "path": path,
                "status": "error"
            })

        ext = Path(file_path).suffix.lower()
        tables = []

        if ext == '.pdf':
            tables = extract_pdf_tables(file_path)
        elif ext in ['.md', '.markdown']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tables = extract_markdown_tables(content)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                tables = extract_markdown_tables(content)
            except Exception:
                return json.dumps({
                    "error": "File format not supported for table extraction",
                    "path": path,
                    "supported_formats": ["pdf", "md", "markdown"],
                    "status": "error"
                })

        if table_index is not None:
            if 1 <= table_index <= len(tables):
                tables = [tables[table_index - 1]]
            else:
                return json.dumps({
                    "error": f"Table index {table_index} out of range (1-{len(tables)})",
                    "total_tables": len(tables),
                    "status": "error"
                })

        if output_format == "markdown":
            md_tables = []
            for table in tables:
                headers = table.get("headers", [])
                rows = table.get("rows", [])

                if headers:
                    md = "| " + " | ".join(str(h) for h in headers) + " |\n"
                    md += "| " + " | ".join("---" for _ in headers) + " |\n"
                    for row in rows:
                        md += "| " + " | ".join(str(c) for c in row) + " |\n"
                    md_tables.append({
                        "table_index": table.get("table_index"),
                        "page": table.get("page"),
                        "markdown": md
                    })
            tables = md_tables

        return json.dumps({
            "tables": tables,
            "total_tables": len(tables),
            "file": file_path,
            "format": ext.lstrip('.'),
            "status": "success"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "path": path,
            "status": "error"
        })


def find_files_impl(
    directory: str,
    name_pattern: Optional[str],
    file_type: Optional[str],
    max_depth: Optional[int],
    size_filter: Optional[str],
    modified_days: Optional[int],
    max_results: int,
    validate_dir_path: Callable[[str], str],
    run_command: Callable[..., Dict[str, Any]],
) -> str:
    """Core logic for find_files tool."""
    try:
        dir_path = validate_dir_path(directory)

        if not os.path.isdir(dir_path):
            return json.dumps({
                "error": f"Directory not found: {dir_path}",
                "directory": directory,
                "status": "error"
            })

        max_results = int(max_results)
        if max_results <= 0:
            max_results = 100

        if max_depth is not None:
            max_depth = int(max_depth)
            if max_depth < 0:
                return json.dumps({"error": "max_depth must be non-negative", "status": "error"})

        if modified_days is not None:
            modified_days = int(modified_days)
            if modified_days < 0:
                return json.dumps({"error": "modified_days must be non-negative", "status": "error"})

        allowed_file_types = {"f", "d", "l", "b", "c", "p", "s"}
        if file_type and file_type not in allowed_file_types:
            return json.dumps({
                "error": f"Invalid file_type: {file_type}. Must be one of: {', '.join(sorted(allowed_file_types))}",
                "status": "error"
            })

        if size_filter and not re.match(r'^[+-]?\d+[bcwkMG]?$', size_filter):
            return json.dumps({
                "error": f"Invalid size_filter: {size_filter}. Expected format: [+-]N[bcwkMG]",
                "status": "error"
            })

        cmd_parts = ["find", shlex.quote(dir_path)]

        if max_depth is not None:
            cmd_parts.append(f"-maxdepth {max_depth}")

        if file_type:
            cmd_parts.append(f"-type {file_type}")

        if name_pattern:
            cmd_parts.append(f"-name {shlex.quote(name_pattern)}")

        if size_filter:
            cmd_parts.append(f"-size {size_filter}")

        if modified_days is not None:
            cmd_parts.append(f"-mtime -{modified_days}")

        cmd = " ".join(cmd_parts) + f" 2>/dev/null | head -n {max_results}"

        result = run_command(cmd, cwd=dir_path)

        if result.get("status") == "error":
            return json.dumps(result)

        output = result.get("stdout", "")
        files = [f.strip() for f in output.split('\n') if f.strip()]

        file_info = []
        for f in files:
            try:
                stat = os.stat(f)
                file_info.append({
                    "path": f,
                    "name": os.path.basename(f),
                    "size_bytes": stat.st_size,
                    "is_dir": os.path.isdir(f)
                })
            except Exception:
                file_info.append({"path": f, "name": os.path.basename(f)})

        return json.dumps({
            "files": file_info,
            "total_files": len(file_info),
            "directory": dir_path,
            "pattern": name_pattern,
            "truncated": len(files) >= max_results,
            "status": "success"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "directory": directory,
            "status": "error"
        })


def transform_text_impl(
    input_text: Optional[str],
    operation: Optional[str],
    expression: Optional[str],
    is_file: bool,
    data_path: str,
    validate_read_path: Callable[[str], str],
    run_command: Callable[..., Dict[str, Any]],
) -> str:
    """Core logic for transform_text tool."""
    if err := validate_required(input_text=input_text, operation=operation, expression=expression):
        return err
    try:
        if operation not in ["sed", "awk", "tr"]:
            return json.dumps({
                "error": f"Invalid operation: {operation}. Use 'sed', 'awk', or 'tr'",
                "status": "error"
            })

        temp_path = None
        if is_file:
            file_path = validate_read_path(input_text)
            if not os.path.isfile(file_path):
                return json.dumps({
                    "error": f"File not found: {file_path}",
                    "status": "error"
                })
            input_source = f"cat {shlex.quote(file_path)}"
        else:
            tmp_dir = os.path.join(os.path.abspath(os.path.expanduser(data_path)), "tmp")
            secure_makedirs(tmp_dir)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir=tmp_dir) as f:
                f.write(input_text)
                temp_path = f.name
            os.chmod(temp_path, 0o600)
            input_source = f"cat {shlex.quote(temp_path)}"

        if operation == "sed":
            cmd = f"{input_source} | sed {shlex.quote(expression)}"
        elif operation == "awk":
            cmd = f"{input_source} | awk {shlex.quote(expression)}"
        elif operation == "tr":
            try:
                tr_args = shlex.split(expression)
                quoted_tr_args = " ".join(shlex.quote(arg) for arg in tr_args)
                cmd = f"{input_source} | tr {quoted_tr_args}"
            except ValueError as e:
                return json.dumps({
                    "error": f"Invalid tr expression: {e}",
                    "status": "error"
                })

        result = run_command(cmd)

        if not is_file and temp_path:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        if result.get("status") == "error":
            return json.dumps(result)

        output = truncate_output(result.get("stdout", ""))

        return json.dumps({
            "output": output,
            "operation": operation,
            "expression": expression,
            "status": "success"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "operation": operation,
            "status": "error"
        })


def get_document_context_impl(
    path: Optional[str],
    query: Optional[str],
    keywords: Optional[str],
    context_chars: int,
    max_sections: int,
    validate_read_path: Callable[[str], str],
) -> str:
    """Core logic for get_document_context tool."""
    if err := validate_required(path=path, query=query):
        return err
    try:
        file_path = validate_read_path(path)

        if not os.path.isfile(file_path):
            return json.dumps({
                "error": f"File not found: {file_path}",
                "path": path,
                "status": "error"
            })

        content, file_format = get_file_content(file_path)

        if file_format == "error" or not content:
            return json.dumps({
                "error": "Failed to read file",
                "path": path,
                "status": "error"
            })

        search_terms = set()

        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
                     'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
                     'what', 'when', 'where', 'which', 'who', 'how', 'this', 'that',
                     'with', 'from', 'they', 'will', 'would', 'there', 'their'}

        for word in re.findall(r'\b\w+\b', query.lower()):
            if len(word) > 3 and word not in stopwords:
                search_terms.add(word)

        if keywords:
            for kw in keywords.split(','):
                kw = kw.strip().lower()
                if kw:
                    search_terms.add(kw)

        if not search_terms:
            search_terms = {w.lower() for w in query.split() if len(w) > 2}

        matches = []
        content_lower = content.lower()

        for term in search_terms:
            for match in re.finditer(re.escape(term), content_lower):
                matches.append({
                    "position": match.start(),
                    "term": term
                })

        matches.sort(key=lambda x: x["position"])

        sections = []
        used_ranges = []

        for match in matches:
            pos = match["position"]

            is_covered = any(start <= pos <= end for start, end in used_ranges)
            if is_covered:
                continue

            start = max(0, pos - context_chars // 2)
            end = min(len(content), pos + context_chars // 2)

            if start > 0:
                sentence_start = content.rfind('.', start - 100, start)
                if sentence_start > start - 100:
                    start = sentence_start + 1

            if end < len(content):
                sentence_end = content.find('.', end, end + 100)
                if sentence_end != -1:
                    end = sentence_end + 1

            section_content = content[start:end].strip()

            section_keywords = [t for t in search_terms if t in section_content.lower()]

            sections.append({
                "content": section_content,
                "relevance_keywords": section_keywords,
                "position": pos,
                "char_range": [start, end]
            })

            used_ranges.append((start, end))

            if len(sections) >= max_sections:
                break

        return json.dumps({
            "sections": sections,
            "total_sections": len(sections),
            "query": query,
            "search_terms": list(search_terms),
            "file": file_path,
            "format": file_format,
            "status": "success"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "path": path,
            "status": "error"
        })
