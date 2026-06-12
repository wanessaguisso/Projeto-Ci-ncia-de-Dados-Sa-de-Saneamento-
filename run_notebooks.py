"""Execute project notebooks in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"
OUTPUT = NOTEBOOKS / "_executed"

NOTEBOOK_LIST = [
    "01_limpeza_dados.ipynb",
    "02_analise_estatistica.ipynb",
    "03_clusterizacao.ipynb",
]


def execute_notebook(name: str) -> None:
    output_path = OUTPUT / name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output",
            output_path.name,
            "--output-dir",
            str(output_path.parent),
            str(NOTEBOOKS / name),
        ],
        check=True,
    )


def main() -> None:
    for name in NOTEBOOK_LIST:
        print(f"▶ Executando: {NOTEBOOKS / name}")
        execute_notebook(name)
        print(f"✅ Salvo: {OUTPUT / name}")


if __name__ == "__main__":
    main()
