"""Executa o notebook com os mesmos algoritmos usados na demonstração."""

import argparse
import json
from pathlib import Path
import sys
import types

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def carregar_funcoes():
    module = types.ModuleType("panorama_notebook")
    sys.modules[module.__name__] = module
    notebook = nbformat.read(ROOT / "panorama.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code" and "definicoes" in cell.metadata.get("tags", []):
            exec(compile(cell.source, "panorama.ipynb", "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=ROOT / "images/v1")
    parser.add_argument("--saida", type=Path, default=ROOT / "outputs")
    parser.add_argument("--resolucao", type=int, default=1600)
    parser.add_argument("--kernel", default="mo446")
    args = parser.parse_args()
    notebook = nbformat.read(ROOT / "panorama.ipynb", as_version=4)
    values = {"entrada": str(args.entrada.resolve()), "saida": str(args.saida.resolve()),
              "resolucao": args.resolucao}
    for cell in notebook.cells:
        if "parametros" in cell.metadata.get("tags", []):
            cell.source = "PARAMETROS = " + repr(values)
    args.saida.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(notebook, timeout=1200, kernel_name=args.kernel,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, args.saida / "panorama_executado.ipynb")
    print(json.dumps({"notebook": str(args.saida / "panorama_executado.ipynb")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
