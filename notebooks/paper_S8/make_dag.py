"""DAG for the TFR + SNe forward model in the joint-S8 PV paper (TikZ/LaTeX).

Manual TikZ layout. Sized for an MNRAS two-column figure
(textwidth ~ 178 mm) via a uniform tikzpicture ``scale``.
"""
import os
import subprocess

OUTPUT_DIR = os.path.expanduser("~/Projects/CANDEL/plots/S8")
TEX_FILE = os.path.join(OUTPUT_DIR, "TFR_DAG.tex")
PDF_FILE = os.path.join(OUTPUT_DIR, "TFR_DAG.pdf")

# =========================================================================
# Manual node positions (x, y) in cm
# =========================================================================
pos = {
    # Row 5 (y=7.0): top-level priors and external data
    'TFR_params':    (1.0,  7.0),
    'eta_prior':     (3.5,  7.0),
    'bias_params':   (6.0,  7.0),
    'density_field': (8.0,  7.0),
    'pqR':           (10.0, 7.0),
    'beta_vext':     (14.5, 7.0),
    # Row 4 (y=5.4): latent samples + deterministic velocity field
    'eta_true':      (3.0,  5.4),
    'r':             (8.5,  5.4),
    'los_vel_r':     (12.0, 5.4),
    # Row 3 (y=3.8): deterministic intermediates
    'M_eta':         (1.5,  3.8),
    'mu_r':          (5.0,  3.8),
    'z_cosmo':       (9.5,  3.8),
    'z_pec':         (13.0, 3.8),
    # Row 2 (y=2.2): predicted observables + side priors
    'sigma_int':     (0.0,  2.2),
    'm_pred':        (3.5,  2.2),
    'z_pred':        (11.5, 2.2),
    'sigma_v':       (15.0, 2.2),
    # Row 1 (y=0.7): observed data
    'eta_obs':       (0.5,  0.7),
    'mag_obs':       (7.0,  0.7),
    'czcmb':         (12.5, 0.7),
    # Row 0 (y=-0.8): main likelihoods
    'eta_ll':        (0.0, -0.8),
    'mag_ll':        (5.5, -0.8),
    'cz_ll':         (12.0, -0.8),
}

latex_labels = {
    'TFR_params':    r'$a_{\rm TFR},\,b_{\rm TFR},\,c_{\rm TFR}$',
    'eta_prior':     r'$\hat{\eta},\,w_\eta$',
    'bias_params':   r'$b_1,\,b_2$',
    'density_field': r'2M{\scriptsize ++}\\density field',
    'pqR':           r'$q,\,R$',
    'los_vel_r':     r'Velocity\\field',
    'beta_vext':     r'$\beta,\,V_{\rm ext}$',
    'eta_true':      r'$\eta_{{\rm true},i}$',
    'r':             r'$r_i$',
    'M_eta':         r'$M_i$',
    'mu_r':          r'$\mu_i$',
    'z_cosmo':       r'$z_{{\rm cosmo},i}$',
    'z_pec':         r'$z_{{\rm pec},i}$',
    'sigma_int':     r'$\sigma_{\rm int}$',
    'm_pred':        r'$m_{{\rm pred},i}$',
    'z_pred':        r'$z_{{\rm pred},i}$',
    'sigma_v':       r'$\sigma_v$',
    'eta_obs':       r'$\eta_{{\rm obs},i}$',
    'mag_obs':       r'$m_{{\rm obs},i}$',
    'czcmb':         r'$z_{{\rm obs},i}$',
    'eta_ll':        r'$\eta_{{\rm obs},i} \mid \eta_{{\rm true},i}$',
    'mag_ll':        r'$m_{{\rm obs},i} \mid m_{{\rm pred},i}$',
    'cz_ll':         r'$z_{{\rm obs},i} \mid z_{{\rm pred},i}$',
}

node_styles = {
    'TFR_params': 'prior', 'eta_prior': 'prior', 'bias_params': 'prior',
    'pqR': 'prior', 'beta_vext': 'prior',
    'eta_true': 'prior', 'r': 'prior',
    'sigma_int': 'prior', 'sigma_v': 'prior',
    'density_field': 'data',
    'eta_obs': 'data', 'mag_obs': 'data', 'czcmb': 'data',
    'los_vel_r': 'det',
    'M_eta': 'det', 'mu_r': 'det', 'z_cosmo': 'det', 'z_pec': 'det',
    'm_pred': 'det', 'z_pred': 'det',
    'eta_ll': 'like', 'mag_ll': 'like', 'cz_ll': 'like',
}

edges = [
    # Distance r and its parents
    ('pqR', 'r'), ('density_field', 'r'), ('bias_params', 'r'),
    # Velocity field as a deterministic function of the density field
    ('density_field', 'los_vel_r'),
    # Redshift branch
    ('r', 'z_cosmo'), ('r', 'z_pec'),
    ('beta_vext', 'z_pec'), ('los_vel_r', 'z_pec'),
    ('z_cosmo', 'z_pred'), ('z_pec', 'z_pred'),
    ('z_pred', 'cz_ll'), ('sigma_v', 'cz_ll'), ('czcmb', 'cz_ll'),
    # Eta branch
    ('eta_prior', 'eta_true'),
    ('eta_true', 'eta_ll'), ('eta_obs', 'eta_ll'),
    # Magnitude branch
    ('r', 'mu_r'),
    ('eta_true', 'M_eta'), ('TFR_params', 'M_eta'),
    ('mu_r', 'm_pred'), ('M_eta', 'm_pred'),
    ('m_pred', 'mag_ll'), ('mag_obs', 'mag_ll'), ('sigma_int', 'mag_ll'),
]

# =========================================================================
# Generate TikZ
# =========================================================================
node_lines = []
for name in pos:
    x, y = pos[name]
    style = node_styles[name]
    label = latex_labels[name]
    node_lines.append(
        f'\\node[{style}] ({name}) at ({x:.2f}, {y:.2f}) {{{label}}};')

edge_lines = []
for a, b in edges:
    key = f'{a}_{b}'
    if key == 'eta_true_eta_ll':
        # Long left-side curve, routed around the magnitude-branch nodes.
        edge_lines.append(
            f'\\draw[dag edge] ({a}) '
            f'.. controls (-1.5, 3.0) and (-1.0, 0.0) .. ({b});')
    else:
        edge_lines.append(f'\\draw[dag edge] ({a}) -- ({b});')

nodes_block = '\n'.join(node_lines)
edges_block = '\n'.join(edge_lines)

x_left = min(p[0] for p in pos.values())
x_right = max(p[0] for p in pos.values())
y_top = max(p[1] for p in pos.values())
legend_y = y_top + 0.8
# Bounding box edges (must match \useasboundingbox below).
bb_left = x_left - 2.0
bb_right = x_right + 1.5
legend_width = 10.0
legend_x0 = 3.5

# Uniform scale applied to all tikz coordinates (and the bounding box) to
# bring the natural ~18.85 cm width down to MNRAS two-column textwidth
# (178 mm). Node text/ellipse sizes are unaffected, so labels stay
# readable.
SCALE = 0.94

tex = rf"""
\documentclass[border=5pt]{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{arrows.meta, shapes.geometric, backgrounds}}
\usepackage{{amsmath, amssymb}}

\definecolor{{colprior}}{{HTML}}{{80A1D4}}
\definecolor{{coldet}}{{HTML}}{{75C9C8}}
\definecolor{{collike}}{{HTML}}{{893168}}
\definecolor{{coldata}}{{HTML}}{{DED9E2}}

\tikzset{{
    dag node/.style={{
        ellipse, draw=black!60, semithick, align=center,
        font=\footnotesize, inner sep=2pt,
    }},
    prior/.style={{dag node, fill=colprior}},
    det/.style={{dag node, fill=coldet}},
    like/.style={{dag node, fill=collike, text=white}},
    data/.style={{dag node, fill=coldata}},
    dag edge/.style={{-{{Stealth[length=3pt, width=2.5pt]}}, thin,
                     draw=black!50}},
}}

\begin{{document}}
\begin{{tikzpicture}}[scale={SCALE}]

\useasboundingbox ({bb_left:.2f}, -1.5) rectangle ({bb_right:.2f}, {legend_y + 0.5:.2f});

% ===== NODES =====
{nodes_block}

% ===== EDGES =====
\begin{{scope}}[on background layer]
{edges_block}
\end{{scope}}

% ===== LEGEND =====
\begin{{scope}}[shift={{({legend_x0:.2f}, {legend_y:.2f})}}]
    \node[prior, minimum width=0.4cm, minimum height=0.25cm,
          font=\footnotesize, inner sep=0pt] (lp) at (0, 0) {{}};
    \node[right, font=\footnotesize] at (lp.east) {{Parameters}};
    \node[det, minimum width=0.4cm, minimum height=0.25cm,
          font=\footnotesize, inner sep=0pt] (ld) at ({0.25 * legend_width:.2f}, 0) {{}};
    \node[right, font=\footnotesize] at (ld.east) {{Deterministic}};
    \node[like, minimum width=0.4cm, minimum height=0.25cm,
          font=\footnotesize, inner sep=0pt] (ll) at ({0.52 * legend_width:.2f}, 0) {{}};
    \node[right, font=\footnotesize] at (ll.east) {{Likelihoods}};
    \node[data, minimum width=0.4cm, minimum height=0.25cm,
          font=\footnotesize, inner sep=0pt] (lo) at ({0.78 * legend_width:.2f}, 0) {{}};
    \node[right, font=\footnotesize] at (lo.east) {{Data}};
\end{{scope}}

\end{{tikzpicture}}
\end{{document}}
"""

# =========================================================================
# Compile
# =========================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(TEX_FILE, "w") as f:
    f.write(tex)

result = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", "-output-directory", OUTPUT_DIR,
     TEX_FILE],
    capture_output=True, text=True,
)

if result.returncode != 0:
    print("pdflatex FAILED:")
    print(result.stdout[-2000:])
    print(result.stderr[-500:])
else:
    for ext in [".aux", ".log"]:
        p = os.path.join(OUTPUT_DIR, f"TFR_DAG{ext}")
        if os.path.exists(p):
            os.remove(p)
    print(f"DAG rendered to {PDF_FILE}")
    subprocess.run(["open", PDF_FILE])
