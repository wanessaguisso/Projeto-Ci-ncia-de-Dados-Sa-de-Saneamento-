"""Run project notebooks in order.

Usage:
  python run_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import papermill as pm


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "_executed"


def run_notebook(input_path: Path, output_path: Path) -> None:
    """Execute a notebook and save the executed copy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(
        input_path=str(input_path),
        output_path=str(output_path),
        kernel_name=None,
        log_output=True,
    )


def main() -> None:
    notebooks = [
        "01_limpeza_dados.ipynb",
        "02_analise_estatistica.ipynb",
        "03_clusterizacao.ipynb",
    ]

    for nb_name in notebooks:
        input_path = NOTEBOOKS_DIR / nb_name
        output_path = OUTPUT_DIR / nb_name
        print(f"▶ Executando: {input_path}")
        run_notebook(input_path, output_path)
        print(f"✅ Salvo: {output_path}")


if __name__ == "__main__":
    main()
