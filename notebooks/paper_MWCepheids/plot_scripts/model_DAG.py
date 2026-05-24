"""DAG for the forward model of MW Cepheid distance ladder (TikZ/LaTeX).

Manual TikZ layout. Sized for an MNRAS two-column figure.
"""
from pathlib import Path
import subprocess


OUTPUT_DIR = Path(__file__).resolve().parent
TEX_FILE = OUTPUT_DIR / "model_DAG.tex"
PDF_FILE = OUTPUT_DIR / "model_DAG.pdf"

# =========================================================================
# Manual node positions (x, y) in cm
# =========================================================================
pos = {
    # Global/population level
    "OHpop": (0.6, 7.0),
    "Ppop": (3.25, 7.0),
    "disk": (5.9, 7.0),
    "PL": (8.65, 7.0),
    "sint": (12.0, 7.0),
    "dpi": (14.35, 7.0),
    "selcuts": (16.85, 7.0),
    # Population distributions
    "OHdist": (0.6, 5.6),
    "Pdist": (3.25, 5.6),
    "posdist": (5.9, 5.6),
    # Per-Cepheid latent parameters
    "OHtrue": (0.6, 4.25),
    "Ptrue": (3.25, 4.25),
    "pos": (5.9, 4.25),
    # Deterministic transforms
    "Mtrue": (8.65, 3.25),
    "mtrue": (11.1, 3.25),
    "pitrue": (14.35, 3.25),
    # Sampling distributions
    "OHsamp": (0.6, 2.05),
    "Psamp": (3.25, 2.05),
    "lbsamp": (5.9, 2.05),
    "msamp": (11.1, 2.05),
    "pisamp": (14.35, 2.05),
    # Observables
    "OHobs": (0.6, 0.65),
    "Pobs": (3.25, 0.65),
    "lbobs": (5.9, 0.65),
    "mobs": (11.1, 0.65),
    "piobs": (14.35, 0.65),
    "selected": (16.7, -0.55),
    "detfrac": (16.85, -1.65),
}

latex_labels = {
    "OHpop": r"$\mu_{[\mathrm{O/H}]},\,\sigma_{[\mathrm{O/H}]}$",
    "Ppop": r"$\mu_{\log P},\,\sigma_{\log P}$",
    "disk": r"Disk\\geometry",
    "PL": r"$M^{W}_{H,1},\,b_W,\,Z_W$",
    "dpi": r"$\delta_\varpi$",
    "sint": r"$\sigma_{\rm int}$",
    "selcuts": r"Selection\\function",
    "detfrac": r"$p(S=1\mid\Lambda)$",
    "OHdist": (
        r"$[\mathrm{O/H}]_i \sim$\\"
        r"$\mathcal{N}(\mu_{[\mathrm{O/H}]},$\\"
        r"$\sigma_{[\mathrm{O/H}]}^2)$"
    ),
    "Pdist": (
        r"$\log P_i \sim$\\"
        r"$\mathcal{N}(\mu_{\log P},$\\"
        r"$\sigma_{\log P}^2)$"
    ),
    "posdist": (
        r"$(d_i,\ell_i,b_i) \sim$\\"
        r"$\pi_{\rm disk}$"
    ),
    "OHtrue": r"$[\mathrm{O/H}]_i$",
    "Ptrue": r"$\log P_i$",
    "pos": r"$d_i,\,\ell_i,\,b_i$",
    "Mtrue": r"$M^W_{H,i}$",
    "mtrue": r"$m^W_{H,i}$",
    "pitrue": r"$\varpi_i$",
    "OHsamp": (
        r"$[\mathrm{O/H}]_{\mathrm{obs},i} \sim$\\"
        r"$\mathcal{N}([\mathrm{O/H}]_i,$\\"
        r"$\epsilon_{[\mathrm{O/H}]}^2)$"
    ),
    "Psamp": r"$\log P_{\mathrm{obs},i} \sim \delta(\log P_i)$",
    "lbsamp": r"$(\ell,b)_{\mathrm{obs},i} \sim \delta((\ell,b)_i)$",
    "msamp": (
        r"$m^W_{H,\mathrm{obs},i} \sim$\\"
        r"$\mathcal{N}(m^W_{H,i},$\\"
        r"$\sigma_{m,i}^2+\sigma_{\rm int}^2)$"
    ),
    "pisamp": (
        r"$\varpi_{\mathrm{obs},i} \sim$\\"
        r"$\mathcal{N}(\varpi_i,\,\sigma_{\varpi,i}^2)$"
    ),
    "OHobs": r"$[\mathrm{O/H}]_{\mathrm{obs},i}$",
    "Pobs": r"$\log P_{\mathrm{obs},i}$",
    "lbobs": r"$\ell_{\mathrm{obs},i},\,b_{\mathrm{obs},i}$",
    "mobs": r"$m^W_{H,\mathrm{obs},i}$",
    "piobs": r"$\varpi_{\mathrm{obs},i}$",
    "selected": r"$S_i=1$",
}

node_styles = {
    "OHpop": "global",
    "Ppop": "global",
    "disk": "global",
    "PL": "global",
    "dpi": "global",
    "sint": "global",
    "selcuts": "global",
    "detfrac": "selection",
    "OHdist": "popdist",
    "Pdist": "popdist",
    "posdist": "popdist",
    "OHtrue": "latent",
    "Ptrue": "latent",
    "pos": "latent",
    "Mtrue": "det",
    "mtrue": "det",
    "pitrue": "det",
    "OHsamp": "sample",
    "Psamp": "sample",
    "lbsamp": "sample",
    "msamp": "sample",
    "pisamp": "sample",
    "OHobs": "data",
    "Pobs": "data",
    "lbobs": "data",
    "mobs": "data",
    "piobs": "data",
    "selected": "conditioned",
}

edges = [
    ("OHpop", "OHdist"),
    ("OHdist", "OHtrue"),
    ("OHtrue", "OHsamp"),
    ("OHsamp", "OHobs"),
    ("OHtrue", "Mtrue"),
    ("Ppop", "Pdist"),
    ("Pdist", "Ptrue"),
    ("Ptrue", "Psamp"),
    ("Psamp", "Pobs"),
    ("Ptrue", "Mtrue"),
    ("disk", "posdist"),
    ("posdist", "pos"),
    ("pos", "lbsamp"),
    ("lbsamp", "lbobs"),
    ("pos", "mtrue"),
    ("pos", "pitrue"),
    ("PL", "Mtrue"),
    ("Mtrue", "mtrue"),
    ("mtrue", "msamp"),
    ("sint", "msamp"),
    ("msamp", "mobs"),
    ("dpi", "pitrue"),
    ("pitrue", "pisamp"),
    ("pisamp", "piobs"),
    ("selcuts", "selected"),
    ("selcuts", "detfrac"),
    ("Pobs", "selected"),
    ("mobs", "selected"),
    ("piobs", "selected"),
    ("lbobs", "selected"),
]

level_labels = [
    ("Global\\\\parameters", -1.15, 7.0),
    ("Population\\\\distribution", -1.15, 5.6),
    ("Latent\\\\parameters", -1.15, 4.25),
    ("Deterministic\\\\transforms", -1.15, 3.25),
    ("Sampling\\\\distribution", -1.15, 2.05),
    ("Observed\\\\data", -1.15, 0.65),
    ("Conditioned\\\\selection", -1.15, -0.55),
]

# =========================================================================
# Generate TikZ
# =========================================================================
node_lines = []
for name in pos:
    x, y = pos[name]
    style = node_styles[name]
    if name == "msamp":
        style = f"{style}, text width=2.35cm"
    label = latex_labels[name]
    node_lines.append(
        f"\\node[{style}] ({name}) at ({x:.2f}, {y:.2f}) {{{label}}};"
    )

edge_lines = []
for a, b in edges:
    key = f"{a}_{b}"
    if key == "OHtrue_Mtrue":
        edge_lines.append(
            "\\draw[dag edge] (OHtrue.south east) "
            ".. controls (2.1, 3.15) and (6.8, 3.0) .. (Mtrue.south west);"
        )
    elif key == "Ptrue_Mtrue":
        edge_lines.append(
            "\\draw[dag edge] (Ptrue.north east) "
            ".. controls (4.6, 4.75) and (7.1, 4.55) .. (Mtrue.north west);"
        )
    elif key == "pos_mtrue":
        edge_lines.append(
            "\\draw[dag edge] (pos.south east) "
            ".. controls (7.2, 2.75) and (9.2, 2.75) .. (mtrue.south west);"
        )
    elif key == "pos_pitrue":
        edge_lines.append(
            "\\draw[dag edge] (pos.north east) "
            ".. controls (7.3, 4.95) and (13.0, 4.95) .. (pitrue.north west);"
        )
    elif key == "sint_msamp":
        edge_lines.append(
            "\\draw[dag edge] (sint.south) "
            ".. controls (12.8, 5.35) and (12.8, 2.55) .. (msamp.north east);"
        )
    elif key == "selcuts_selected":
        edge_lines.append(
            "\\draw[sel edge] (selcuts.south) "
            ".. controls (16.8, 4.0) and (16.8, 0.6) .. (selected.north);"
        )
    elif key == "selcuts_detfrac":
        edge_lines.append(
            "\\draw[sel edge] (selcuts.south east) "
            ".. controls (18.3, 4.5) and (18.3, -1.65) .. (detfrac.east);"
        )
    elif key == "Pobs_selected":
        edge_lines.append(
            "\\draw[sel edge] (Pobs.south east) "
            ".. controls (5.2, -0.55) and (14.1, -0.55) .. (selected.west);"
        )
    elif key == "lbobs_selected":
        edge_lines.append(
            "\\draw[sel edge] (lbobs.south east) "
            ".. controls (8.0, -0.70) and (14.2, -0.70) .. (selected.west);"
        )
    elif key == "mobs_selected":
        edge_lines.append(
            "\\draw[sel edge] (mobs.south east) "
            ".. controls (12.3, -0.40) and (14.7, -0.40) .. (selected.west);"
        )
    elif key == "piobs_selected":
        edge_lines.append(
            "\\draw[sel edge] (piobs.south east) "
            ".. controls (15.1, -0.30) and (15.6, -0.55) .. (selected.west);"
        )
    else:
        edge_lines.append(f"\\draw[dag edge] ({a}) -- ({b});")

level_lines = [
    (
        f"\\node[level label, anchor=east, align=right] at ({x:.2f}, {y:.2f})"
        f" {{{label}}};"
    )
    for label, x, y in level_labels
]

nodes_block = "\n".join(node_lines)
edges_block = "\n".join(edge_lines)
levels_block = "\n".join(level_lines)
group_edge = ""

x_left = min(p[0] for p in pos.values())
x_right = max(p[0] for p in pos.values())
y_top = max(p[1] for p in pos.values())
bb_left = x_left - 3.9
bb_right = x_right + 2.05
bb_top = y_top + 0.85
bb_bottom = -2.25

tex = rf"""
\documentclass[border=5pt]{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{arrows.meta, backgrounds, decorations.pathreplacing, fit, shapes.geometric}}
\usepackage{{amsmath, amssymb}}

\definecolor{{colglobal}}{{HTML}}{{4F6D7A}}
\definecolor{{colpop}}{{HTML}}{{8F5A83}}
\definecolor{{coldet}}{{HTML}}{{1F9D8A}}
\definecolor{{colsample}}{{HTML}}{{8A4F7D}}
\definecolor{{coldata}}{{HTML}}{{D8D8D8}}

\tikzset{{
    dag node/.style={{
        draw=black!70, semithick, align=center,
        font=\scriptsize, inner sep=2.3pt,
    }},
    global/.style={{dag node, rectangle, rounded corners=2pt,
        minimum width=1.75cm, minimum height=0.56cm, fill=colglobal!10}},
    selection/.style={{dag node, rectangle, dashed, draw=black!60,
        fill=white, minimum width=1.85cm, minimum height=0.56cm}},
    popdist/.style={{dag node, rectangle, draw=colpop!85!black,
        dashed, fill=colpop!8, minimum height=0.66cm,
        text width=1.95cm, inner sep=1.3pt}},
    latent/.style={{dag node, ellipse, fill=white,
        minimum width=1.45cm, minimum height=0.58cm}},
    det/.style={{dag node, diamond, aspect=2.35, draw=coldet!85!black,
        fill=coldet!8, inner xsep=1pt, inner ysep=1pt}},
    sample/.style={{dag node, rectangle, draw=colsample!85!black,
        fill=colsample!8, minimum height=0.62cm,
        text width=1.95cm, inner sep=1.3pt}},
    data/.style={{dag node, rectangle, very thick, fill=coldata,
        minimum width=1.5cm, minimum height=0.56cm}},
    conditioned/.style={{data, double, double distance=1pt}},
    plate/.style={{draw=black!55, rounded corners=5pt, dashed,
        inner xsep=9pt, inner ysep=10pt}},
    level label/.style={{font=\scriptsize, text=black!70}},
    legend text/.style={{font=\scriptsize, anchor=west}},
    dag edge/.style={{-{{Stealth[length=3pt, width=2.5pt]}}, thin,
        draw=black!55}},
    sel edge/.style={{dag edge, draw=black!38}},
}}

\begin{{document}}
\begin{{tikzpicture}}

\useasboundingbox ({bb_left:.2f}, {bb_bottom:.2f}) rectangle ({bb_right:.2f}, {bb_top:.2f});

% ===== LEVEL LABELS =====
{levels_block}

% ===== NODES =====
{nodes_block}

% ===== PLATE =====
\begin{{scope}}[on background layer]
    \node[plate, fit=(OHdist)(Pdist)(posdist)(OHobs)(Pobs)(lbobs)(mobs)(piobs)(selected),
          label={{[font=\scriptsize, anchor=north east]north east: Cepheid $i=1,\ldots,N_p$}}] {{}};
\end{{scope}}

% ===== GLOBAL PARAMETER GROUP =====
\draw[decorate, decoration={{brace, amplitude=4pt}}, draw=black!45]
    (OHpop.north west) -- node[above=4pt, font=\scriptsize] (LambdaBrace) {{$\Lambda$}} (selcuts.north east);

% ===== EDGES =====
\begin{{scope}}[on background layer]
{edges_block}
{group_edge}
\end{{scope}}

\end{{tikzpicture}}
\end{{document}}
"""

# =========================================================================
# Compile
# =========================================================================
TEX_FILE.write_text(tex)

result = subprocess.run(
    [
        "pdflatex",
        "-interaction=nonstopmode",
        "-output-directory",
        str(OUTPUT_DIR),
        str(TEX_FILE),
    ],
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print("pdflatex FAILED:")
    print(result.stdout[-2000:])
    print(result.stderr[-500:])
else:
    for ext in [".aux", ".log"]:
        p = OUTPUT_DIR / f"model_DAG{ext}"
        if p.exists():
            p.unlink()
    print(f"DAG rendered to {PDF_FILE}")
