"""Shared helpers for notebook builders."""
from __future__ import annotations

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


def write_notebook(path: str, cells: list) -> None:
    nb = new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"wrote {path}  ({len(cells)} cells)")
