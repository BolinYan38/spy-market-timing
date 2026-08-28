"""Run lightweight repository checks that do not require the raw dataset."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def validate_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    if notebook.get("nbformat") != 4:
        raise ValueError("Expected a Jupyter notebook using nbformat 4")

    cell_ids = [cell.get("id") for cell in cells if cell.get("id")]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Notebook contains duplicate cell IDs")

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    error_cells: list[int] = []

    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        try:
            ast.parse(source, filename=f"{path}:cell-{index}")
        except SyntaxError as exc:
            raise SyntaxError(f"Invalid Python syntax in notebook cell {index}") from exc

        if any(output.get("output_type") == "error" for output in cell.get("outputs", [])):
            error_cells.append(index)

    if error_cells:
        raise ValueError(f"Notebook contains saved error outputs in cells: {error_cells}")

    print(
        f"Validated {path}: {len(cells)} cells, "
        f"{len(code_cells)} code cells, no syntax or saved-output errors."
    )


if __name__ == "__main__":
    notebook_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "notebooks/spy-analysis.ipynb"
    )
    validate_notebook(notebook_path)
