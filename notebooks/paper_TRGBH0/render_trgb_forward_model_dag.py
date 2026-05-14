"""Render the TRGB forward-model DAG for the TRGBH0 paper.

Manual TikZ layout. Sized for an MNRAS two-column figure.
"""
from pathlib import Path
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
TEX_FILE = SCRIPT_DIR / "trgb_forward_model_dag.tex"
PDF_FILE = OUTPUT_DIR / "trgb_forward_model_dag.pdf"

# =========================================================================
# Manual node positions (x, y) in cm
# =========================================================================
pos = {
    # Global/model-level quantities
    "Mtrgb": (0.65, 8.20),
    "cstar": (2.15, 8.20),
    "cpars": (3.85, 8.20),
    "sigint": (5.55, 8.20),
    "H0": (7.25, 8.20),
    "rho": (8.95, 8.20),
    "Vfield": (10.65, 8.20),
    "bias": (12.35, 8.20),
    "pv": (14.05, 8.20),
    "sigv": (15.75, 8.20),
    "selcuts": (17.45, 8.20),
    # Anchor constraints
    "anchprior": (-0.25, 7.05),
    "anchmu": (-0.25, 6.30),
    "anchgeom": (-0.25, 5.45),
    "anchmtrue": (2.45, 5.45),
    "anchsky": (-0.25, 4.25),
    "anchmobs": (2.45, 4.25),
    # Host-level distance and PV model
    "skydelta": (5.05, 3.55),
    "rdist": (8.75, 3.55),
    "rtrue": (8.75, 2.45),
    "mu": (6.45, 1.40),
    "zcos": (8.75, 1.40),
    "vpec": (11.15, 1.40),
    # Host-level colour and observables
    "cdist": (2.25, 3.15),
    "ctrue": (2.25, 2.05),
    "csamp": (1.65, -0.65),
    "cobs": (2.25, -1.95),
    "mtrue": (5.45, 0.45),
    "cztrue": (11.15, 0.40),
    "msamp": (5.45, -0.65),
    "czsamp": (11.15, -0.65),
    "mobs": (4.75, -1.95),
    "czobs": (11.15, -1.95),
    "selected": (14.70, -1.95),
    "detfrac": (14.70, -3.05),
}

latex_labels = {
    "Mtrgb": r"$M_{\rm TRGB}$",
    "cstar": r"$c_\star$",
    "cpars": r"$(\bar c,\sigma_c)$",
    "sigint": r"$\sigma_{\rm int}$",
    "H0": r"$H_0$",
    "pv": r"$\mathbf{V}_{\rm ext}$",
    "sigv": r"$\sigma_v$",
    "rho": r"$\rho(\bm{r})$",
    "Vfield": r"$\bm{V}(\bm{r})$",
    "bias": r"$\bm{b}$",
    "selcuts": r"TRGB\\selection",
    "anchprior": r"$\mu_a \sim p(\mu_a)$",
    "anchmu": r"$\mu_a$",
    "anchgeom": (
        r"$\mu_{a,\rm obs} \sim$\\"
        r"$\mathcal{N}(\mu_a,\epsilon_{\mu,a}^2)$"
    ),
    "anchmtrue": r"$m_a$",
    "anchsky": (
        r"$(\ell_a,b_a) \sim$\\"
        r"$\delta(\ell_a-\ell_{{\rm obs},a})$\\"
        r"$\delta(b_a-b_{{\rm obs},a})$"
    ),
    "anchmobs": (
        r"$m_{{\rm obs},a} \sim$\\"
        r"$\mathcal{N}(m_a,$\\"
        r"$\epsilon_{m,a}^2)$"
    ),
    "skydelta": (
        r"$(\ell_i,b_i) \sim$\\"
        r"$\delta(\ell_i-\ell_{{\rm obs},i})$\\"
        r"$\delta(b_i-b_{{\rm obs},i})$"
    ),
    "rdist": (
        r"$(r_i,\ell_i,b_i) \sim$\\"
        r"$p(\mathbf{x}\mid\rho,\bm{b},H_0)$"
    ),
    "rtrue": r"$r_i$",
    "mu": r"$\mu_i$",
    "zcos": r"$z_{{\rm cos},i}$",
    "vpec": r"$V_{{\rm pec},i}$",
    "cdist": (
        r"$c_i \sim$\\"
        r"$\mathcal{N}(\bar c,\sigma_c^2)$"
    ),
    "ctrue": r"$c_i$",
    "csamp": (
        r"$c_i^{\rm obs} \sim$\\"
        r"$\mathcal{N}(c_i,\epsilon_{c,i}^2)$"
    ),
    "cobs": r"$c_i^{\rm obs}$",
    "mtrue": r"$m_i$",
    "cztrue": r"$cz_i$",
    "msamp": (
        r"$m_{{\rm obs},i} \sim$\\"
        r"$\mathcal{N}(m_i,\epsilon_{m,i}^2+\sigma_{\rm int}^2)$"
    ),
    "czsamp": (
        r"$cz_{{\rm CMB},i} \sim$\\"
        r"$\mathcal{N}(cz_i,\epsilon_{cz,i}^2+\sigma_v^2)$"
    ),
    "mobs": r"$m_{{\rm obs},i}$",
    "czobs": r"$cz_{{\rm CMB},i}$",
    "selected": r"$S_i=1$",
    "detfrac": r"$p(S=1\mid\Lambda)$",
}

node_styles = {
    "Mtrgb": "global",
    "cstar": "global",
    "cpars": "global",
    "sigint": "global",
    "H0": "global",
    "pv": "global",
    "sigv": "global",
    "rho": "input",
    "Vfield": "input",
    "bias": "global",
    "selcuts": "global",
    "anchprior": "popdist",
    "anchmu": "latent",
    "anchgeom": "sample",
    "anchmtrue": "det",
    "anchsky": "sample",
    "anchmobs": "data",
    "skydelta": "sample",
    "rdist": "popdist",
    "rtrue": "latent",
    "mu": "det",
    "zcos": "det",
    "vpec": "det",
    "cdist": "sample",
    "ctrue": "latent",
    "csamp": "sample",
    "cobs": "data",
    "mtrue": "det",
    "cztrue": "det",
    "msamp": "sample",
    "czsamp": "sample",
    "mobs": "data",
    "czobs": "data",
    "selected": "conditioned",
    "detfrac": "selection",
}

edges = [
    ("Mtrgb", "anchmtrue"),
    ("cpars", "cdist"),
    ("cdist", "ctrue"),
    ("ctrue", "csamp"),
    ("csamp", "cobs"),
    ("anchprior", "anchmu"),
    ("anchmu", "anchgeom"),
    ("anchmu", "anchmtrue"),
    ("anchmtrue", "anchmobs"),
    ("anchsky", "anchmobs"),
    ("rho", "rdist"),
    ("bias", "rdist"),
    ("H0", "rdist"),
    ("rdist", "skydelta"),
    ("rdist", "rtrue"),
    ("rtrue", "mu"),
    ("rtrue", "zcos"),
    ("H0", "zcos"),
    ("Vfield", "vpec"),
    ("pv", "vpec"),
    ("skydelta", "vpec"),
    ("rtrue", "vpec"),
    ("Mtrgb", "mtrue"),
    ("cstar", "mtrue"),
    ("ctrue", "mtrue"),
    ("mu", "mtrue"),
    ("mtrue", "msamp"),
    ("sigint", "msamp"),
    ("msamp", "mobs"),
    ("zcos", "cztrue"),
    ("vpec", "cztrue"),
    ("cztrue", "czsamp"),
    ("sigv", "czsamp"),
    ("czsamp", "czobs"),
    ("selcuts", "selected"),
    ("mobs", "selected"),
    ("selcuts", "detfrac"),
]

level_labels = []


def node_style(name):
    style = node_styles[name]
    if name in {"msamp", "czsamp"}:
        style = f"{style}, text width=3.35cm"
    if name in {"cdist", "csamp"}:
        style = f"{style}, text width=2.15cm"
    if name in {"skydelta"}:
        style = f"{style}, text width=2.55cm"
    if name in {"anchprior", "anchgeom", "anchsky", "anchmobs", "rdist"}:
        style = f"{style}, text width=2.20cm"
    if name == "sigv":
        style = f"{style}, minimum width=1.05cm"
    if name == "sigint":
        style = f"{style}, minimum width=1.15cm"
    return style


# =========================================================================
# Generate TikZ
# =========================================================================
node_lines = []
for name in pos:
    x, y = pos[name]
    node_lines.append(
        f"\\node[{node_style(name)}] ({name}) at "
        f"({x:.2f}, {y:.2f}) {{{latex_labels[name]}}};"
    )

edge_lines = []
for a, b in edges:
    key = f"{a}_{b}"
    if key == "Mtrgb_anchmtrue":
        edge_lines.append(
            "\\draw[dag edge] (Mtrgb.south east) "
            ".. controls (1.45, 7.1) and (1.85, 5.95) .. (anchmtrue.north);"
        )
    elif key == "cpars_cdist":
        edge_lines.append(
            "\\draw[dag edge] (cpars.south) "
            ".. controls (4.15, 6.30) and (3.55, 4.10) .. (cdist.north east);"
        )
    elif key == "Mtrgb_mtrue":
        edge_lines.append(
            "\\draw[dag edge] (Mtrgb.south east) "
            ".. controls (2.10, 5.00) and (3.65, 1.10) .. (mtrue.north west);"
        )
    elif key == "cstar_mtrue":
        edge_lines.append(
            "\\draw[dag edge] (cstar.south east) "
            ".. controls (3.00, 5.65) and (4.05, 1.35) .. (mtrue.north);"
        )
    elif key == "ctrue_mtrue":
        edge_lines.append(
            "\\draw[dag edge] (ctrue.south east) "
            ".. controls (3.25, 1.50) and (4.55, 0.95) .. (mtrue.north west);"
        )
    elif key == "sigint_msamp":
        edge_lines.append(
            "\\draw[dag edge] (sigint.south) "
            ".. controls (5.55, 5.75) and (6.25, 0.10) .. ([xshift=1.05cm]msamp.north);"
        )
    elif key == "rho_rdist":
        edge_lines.append(
            "\\draw[dag edge] (rho.south) "
            ".. controls (7.15, 6.85) and (8.75, 4.25) .. (rdist.north);"
        )
    elif key == "bias_rdist":
        edge_lines.append(
            "\\draw[dag edge] (bias.south) "
            ".. controls (10.30, 6.10) and (9.20, 4.25) .. (9.00, 3.88);"
        )
    elif key == "H0_rdist":
        edge_lines.append(
            "\\draw[dag edge] (H0.south) "
            ".. controls (5.20, 6.25) and (8.30, 4.25) .. (8.50, 3.88);"
        )
    elif key == "H0_zcos":
        edge_lines.append(
            "\\draw[dag edge] (H0.south) "
            ".. controls (5.20, 5.00) and (8.05, 2.05) .. (zcos.north west);"
        )
    elif key == "Vfield_vpec":
        edge_lines.append(
            "\\draw[dag edge] (Vfield.south) "
            ".. controls (8.60, 6.10) and (10.6, 2.40) .. (vpec.north west);"
        )
    elif key == "pv_vpec":
        edge_lines.append(
            "\\draw[dag edge] (pv.south) "
            ".. controls (12.10, 5.00) and (12.30, 1.55) .. (vpec.east);"
        )
    elif key == "skydelta_vpec":
        edge_lines.append(
            "\\draw[dag edge] (skydelta.east) "
            ".. controls (7.0, 3.25) and (10.20, 1.65) .. (vpec.west);"
        )
    elif key == "rdist_skydelta":
        continue
    elif key == "rtrue_vpec":
        edge_lines.append(
            "\\draw[dag edge] (rtrue.east) "
            ".. controls (9.65, 2.35) and (10.75, 2.20) .. (vpec.north);"
        )
    elif key == "sigv_czsamp":
        edge_lines.append(
            "\\draw[dag edge] (sigv.south) "
            ".. controls (13.75, 4.10) and (12.65, -0.65) .. (czsamp.east);"
        )
    elif key == "selcuts_selected":
        edge_lines.append(
            "\\draw[sel edge] (selcuts.south) "
            ".. controls (14.95, 5.05) and (14.70, -0.20) .. (selected.north);"
        )
    elif key == "mobs_selected":
        edge_lines.append(
            "\\draw[sel edge] (mobs.south east) "
            ".. controls (7.20, -2.65) and (12.00, -2.65) .. (selected.south west);"
        )
    elif key == "selcuts_detfrac":
        edge_lines.append(
            "\\draw[sel edge] (selcuts.south east) "
            ".. controls (17.75, 5.05) and (17.20, -3.05) .. (detfrac.east);"
        )
    else:
        edge_lines.append(f"\\draw[dag edge] ({a}) -- ({b});")

level_lines = [
    (
        f"\\node[level label, anchor=east, align=right] "
        f"at ({x:.2f}, {y:.2f}) {{{label}}};"
    )
    for label, x, y in level_labels
]

nodes_block = "\n".join(node_lines)
edges_block = "\n".join(edge_lines)
levels_block = "\n".join(level_lines)

x_left = min(p[0] for p in pos.values())
x_right = max(p[0] for p in pos.values())
y_top = max(p[1] for p in pos.values())
bb_left = x_left - 2.00
bb_right = x_right + 1.75
bb_top = y_top + 1.80
bb_bottom = -3.40

tex = rf"""
\documentclass[border=5pt]{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{arrows.meta, backgrounds, decorations.pathreplacing, fit, shapes.geometric}}
\usepackage{{amsmath, amssymb}}
\usepackage{{bm}}

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
        minimum width=1.45cm, minimum height=0.56cm, fill=colglobal!10}},
    input/.style={{dag node, rectangle, rounded corners=2pt,
        dashed, draw=black!60, minimum width=1.45cm,
        minimum height=0.56cm, fill=coldata!35}},
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
    dag edge/.style={{-{{Stealth[length=3pt, width=2.5pt]}}, thin,
        draw=black!55}},
    sel edge/.style={{dag edge, draw=black!38}},
}}

\begin{{document}}
\begin{{tikzpicture}}

\useasboundingbox ({bb_left:.2f}, {bb_bottom:.2f}) rectangle ({bb_right:.2f}, {bb_top:.2f});

% ===== LEGEND =====
\begin{{scope}}[on background layer]
    \fill[black!6, rounded corners=3pt] (-1.35, 9.02) rectangle (18.25, 9.82);
\end{{scope}}
\node[global, minimum width=1.35cm] at (-0.55, 9.42) {{Global}};
\node[input, minimum width=1.55cm] at (1.45, 9.42) {{Fixed\\input}};
\node[sample, text width=1.35cm] at (3.45, 9.42) {{Sampling}};
\node[latent, minimum width=1.25cm] at (5.20, 9.42) {{Latent}};
\node[det, minimum width=1.25cm] at (7.15, 9.42) {{Deterministic}};
\node[data, minimum width=1.35cm] at (9.55, 9.42) {{Observed}};
\node[conditioned, minimum width=1.35cm] at (11.50, 9.42) {{Selection}};
\node[popdist, text width=1.50cm] at (13.65, 9.42) {{Population}};
\node[selection, minimum width=1.65cm] at (15.80, 9.42) {{Detection\\fraction}};

% ===== NODES =====
{nodes_block}

% ===== PLATES =====
\begin{{scope}}[on background layer]
    \node[plate, fit=(anchprior)(anchmu)(anchgeom)(anchmtrue)(anchsky)(anchmobs)] {{}};
    \node[plate, inner ysep=6pt, fit=(skydelta)(rdist)(rtrue)(mu)(zcos)(vpec)(cdist)(ctrue)(csamp)(cobs)(mtrue)(cztrue)(msamp)(czsamp)(mobs)(czobs)(selected)] {{}};
\end{{scope}}

% ===== EDGES =====
\begin{{scope}}[on background layer]
{edges_block}
\end{{scope}}
\draw[dag edge, draw=black!70]
    (rdist.west) -- (skydelta.east);

% ===== PLATE LABELS =====
\node[font=\scriptsize, fill=white, inner sep=1.2pt, anchor=west]
    at (-1.16, 7.56) {{Anchor $a\in\{{\rm LMC,N4258\}}$}};
\node[font=\scriptsize, fill=white, inner sep=1.2pt, anchor=east]
    at (15.70, 3.95) {{TRGB host $i=1,\ldots,N_{{\rm host}}$}};

\end{{tikzpicture}}
\end{{document}}
"""

# =========================================================================
# Compile
# =========================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
        p = OUTPUT_DIR / f"trgb_forward_model_dag{ext}"
        if p.exists():
            p.unlink()
    print(f"DAG rendered to {PDF_FILE}")
