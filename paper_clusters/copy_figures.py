"""Copy figures and tables to the Overleaf project folder.

Tables are copied without the LaTeX preamble so they can be \\input{}'d.
"""
from __future__ import annotations

from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_SRC = PROJECT_ROOT / "paper_clusters" / "figures"
TABLES_SRC = PROJECT_ROOT / "paper_clusters" / "tables"

OVERLEAF_ROOT = Path(
    "/Users/yasin/Dropbox/Apps/Overleaf/Cluster Anisotropies !"
)
FIGURES_DST = OVERLEAF_ROOT / "Figures"
TABLES_DST = OVERLEAF_ROOT / "Tables"


def strip_preamble(tex: str) -> str:
    """Extract only the tabular environment from the document.

    Removes document preamble, table* wrapper, caption, label, etc.
    Only keeps content from \\begin{tabular} to \\end{tabular}.
    """
    # First remove document wrapper if present
    begin = tex.find(r"\begin{document}")
    end = tex.find(r"\end{document}")
    if begin != -1 and end != -1 and end > begin:
        tex = tex[begin + len(r"\begin{document}") : end]

    # Extract only the tabular environment
    tabular_begin = tex.find(r"\begin{tabular")
    tabular_end = tex.rfind(r"\end{tabular}")

    if tabular_begin != -1 and tabular_end != -1:
        # Include the \end{tabular} itself
        tabular_end = tabular_end + len(r"\end{tabular}")
        tex = tex[tabular_begin:tabular_end]

    return tex.strip() + "\n"


def copy_figures() -> None:
    """Copy all figure files to the Overleaf Figures folder."""
    FIGURES_DST.mkdir(parents=True, exist_ok=True)
    for path in FIGURES_SRC.iterdir():
        if path.is_file():
            shutil.copy2(path, FIGURES_DST / path.name)


def copy_tables() -> None:
    """Copy table .tex files without preambles to the Overleaf Tables folder."""
    TABLES_DST.mkdir(parents=True, exist_ok=True)
    for path in TABLES_SRC.glob("*.tex"):
        content = path.read_text()
        stripped = strip_preamble(content)
        (TABLES_DST / path.name).write_text(stripped)


def main() -> None:
    copy_figures()
    copy_tables()


if __name__ == "__main__":
    main()
