"""Data import/export for AxiomPy.

Supports four formats without external libraries:

- ``.axi`` — AxiomPy's custom binary/text format (JSON-based with version header)
- ``.csv`` — Comma-separated values
- ``.json`` — JavaScript Object Notation
- ``.txt`` — Pretty-printed text tables

Accessible via ``Axiom.io``.

Examples:
    >>> Axiom.io.save(data, "data.axi")
    >>> loaded = Axiom.io.load("data.axi")
"""

import csv
import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# .axi format (custom AxiomPy format)
# ---------------------------------------------------------------------------

def _axi_header():
    return {"__axiompy__": True, "version": "1.0", "format": "axi"}


def _is_axi_file(path: str) -> bool:
    """Check if file is a valid .axi format."""
    try:
        with open(path) as f:
            first = f.read(200)
        if '"__axiompy__"':  # check without parsing
            pass
        data = json.loads(first.rstrip("\n") + "\n}")
        return data.get("__axiompy__") is True
    except Exception:
        return False


def save_axi(data: Any, path: str, pretty: bool = True) -> None:
    """Save data to a custom .axi format file.

    The format wraps JSON with a version header for future compatibility.

    Args:
        data: Any JSON-serializable data (list, dict, number, string, etc.).
        path: Output file path (.axi extension recommended).
        pretty: If True, human-readable indented output (default True).
    """
    package = {"__axiompy__": True, "version": "1.0", "format": "axi", "data": data}
    with open(path, "w") as f:
        json.dump(package, f, indent=2 if pretty else None)
    # Ensure trailing newline
    with open(path, "a") as f:
        f.write("\n")


def load_axi(path: str) -> Any:
    """Load data from a .axi format file.

    Args:
        path: Input file path.

    Returns:
        The stored data (type depends on what was saved).
    """
    with open(path) as f:
        package = json.load(f)
    if not package.get("__axiompy__"):
        raise ValueError(f"Not a valid .axi file: {path}")
    return package["data"]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(data: list[list[Any]], path: str, headers: list[str] | None = None) -> None:
    """Save tabular data to a CSV file.

    Args:
        data: List of rows, each a list of values.
        path: Output file path.
        headers: Optional list of column header names.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        for row in data:
            writer.writerow(row)


def load_csv(path: str, has_headers: bool = False) -> list[list[Any]] | tuple[list[str], list[list[Any]]]:
    """Load data from a CSV file.

    Args:
        path: Input file path.
        has_headers: If True, the first row is treated as a header and returned separately.

    Returns:
        List of rows (each a list of strings). If has_headers is True,
        returns (headers, rows) as a tuple.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    if has_headers and rows:
        return rows[0], rows[1:]
    return rows


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_json(data: Any, path: str, pretty: bool = True) -> None:
    """Save data to a JSON file.

    Args:
        data: Any JSON-serializable data.
        path: Output file path.
        pretty: If True, human-readable indented output (default True).
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2 if pretty else None)
    with open(path, "a") as f:
        f.write("\n")


def load_json(path: str) -> Any:
    """Load data from a JSON file.

    Args:
        path: Input file path.

    Returns:
        Parsed JSON data.
    """
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# TXT (pretty-print tables)
# ---------------------------------------------------------------------------

def save_txt(data: list[list[Any]], path: str, headers: list[str] | None = None) -> None:
    """Save tabular data as a pretty-printed text file.

    Args:
        data: List of rows, each a list of values.
        path: Output file path.
        headers: Optional list of column header names.
    """
    all_rows = (headers or []) + [row for row in data] if headers else data
    if not all_rows:
        with open(path, "w") as f:
            f.write("(empty)\n")
        return

    col_widths = []
    for col_idx in range(len(all_rows[0])):
        widths = []
        for row in all_rows:
            val = str(row[col_idx]) if col_idx < len(row) else ""
            widths.append(len(val))
        col_widths.append(max(widths) if widths else 0)

    lines = []
    for row_idx, row in enumerate(all_rows):
        parts = []
        for col_idx in range(len(row)):
            val = str(row[col_idx])
            parts.append(val.ljust(col_widths[col_idx]))
        lines.append("  ".join(parts))
        if headers and row_idx == 0:
            lines.append("-" * sum(col_widths + [2 * (len(col_widths) - 1)]))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_txt(path: str) -> list[str]:
    """Load a text file as a list of lines.

    Args:
        path: Input file path.

    Returns:
        List of stripped lines.
    """
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


# ---------------------------------------------------------------------------
# Auto-detect loader
# ---------------------------------------------------------------------------

def load(path: str) -> Any:
    """Auto-detect format and load data from *path*.

    Supported extensions: ``.axi``, ``.json``, ``.csv``, ``.txt``.

    Args:
        path: Input file path.

    Returns:
        Loaded data. For .csv returns list of rows; for .txt returns list of lines.
    """
    ext = Path(path).suffix.lower()
    if ext == ".axi":
        return load_axi(path)
    elif ext == ".json":
        return load_json(path)
    elif ext == ".csv":
        return load_csv(path)
    elif ext == ".txt":
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def save(data: Any, path: str, **kwargs) -> None:
    """Save data to *path*, auto-detecting format from the extension.

    Supported extensions: ``.axi``, ``.json``, ``.csv``, ``.txt``.

    Args:
        data: Data to save (list of rows for .csv/.txt).
        path: Output file path.
        **kwargs: Passed to the format-specific save function.
    """
    ext = Path(path).suffix.lower()
    if ext == ".axi":
        save_axi(data, path, **kwargs)
    elif ext == ".json":
        save_json(data, path, **kwargs)
    elif ext == ".csv":
        headers = kwargs.get("headers")
        save_csv(data, path, headers=headers)
    elif ext == ".txt":
        headers = kwargs.get("headers")
        save_txt(data, path, headers=headers)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ---------------------------------------------------------------------------
# Facade wrapper
# ---------------------------------------------------------------------------

class AxiomIO:
    """Import and export data in multiple formats.

    Supported formats:
        - ``.axi`` — AxiomPy custom format (JSON-based with version header)
        - ``.json`` — Standard JSON
        - ``.csv``  — Comma-separated values
        - ``.txt``  — Pretty-printed text table
    """

    save_axi = staticmethod(save_axi)
    load_axi = staticmethod(load_axi)
    save_csv = staticmethod(save_csv)
    load_csv = staticmethod(load_csv)
    save_json = staticmethod(save_json)
    load_json = staticmethod(load_json)
    save_txt = staticmethod(save_txt)
    load_txt = staticmethod(load_txt)
    save = staticmethod(save)
    load = staticmethod(load)

    def help(self) -> str:
        return (
            "=== AxiomPy Data I/O Help ===\n"
            "\n"
            "Save data:\n"
            "  Axiom.io.save(data, 'file.axi')    # auto-detect format\n"
            "  Axiom.io.save_csv(rows, 'out.csv', headers=['a','b'])\n"
            "  Axiom.io.save_json(obj, 'out.json')\n"
            "  Axiom.io.save_txt(rows, 'out.txt', headers=['a','b'])\n"
            "\n"
            "Load data:\n"
            "  data = Axiom.io.load('file.axi')     # auto-detect\n"
            "  rows = Axiom.io.load_csv('data.csv')\n"
            "  obj  = Axiom.io.load_json('data.json')\n"
            "  lines = Axiom.io.load_txt('notes.txt')\n"
        )
