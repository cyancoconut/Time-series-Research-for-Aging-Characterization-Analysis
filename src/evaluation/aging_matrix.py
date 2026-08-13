"""Build the fleet-wide aging matrix (Alterungsmatrix) for evaluation.

Port of the exploratory `alterungsmatrix.ipynb` notebook. Aggregates per-cell
capacity loss, normalized by Ah throughput, into a matrix indexed by
(C_Rate, Temperature, DOD, SOC) and renders it as interactive plots. The HTML
also leads with a fleet capacity-fade plot (normalized capacity vs equivalent
full cycle, faceted by DOD) ported from the `alterungsmatrix.ipynb` notebook.

Per cell:
    capacity_lost       = max - min of Capacity_py across the cell's check-ups
    Delta_Ah_throughput = max - min of Ah_throughput across the cell's check-ups

Per (C_Rate, Temperature, DOD, SOC) group:
    mean/std of both, a cell count, and the normalized loss
    capacity_lost_norm = capacity_lost_mean / Delta_Ah_throughput_mean

Inputs (driven by `download_from`): the same fleet capacity table that
`export_cap_pulse.py` builds from `40_capacity_monitore/*_capacity.csv`. Those
CSVs must carry an `Ah_throughput` column (added by `output/export_capacity.py`);
older runs without it need the pipeline re-run first.

Outputs (driven by `upload_to`):
    <working_path>/50_evaluation/aging_matrix.csv
    <working_path>/50_evaluation/aging_matrix.html
    MinIO: <minio_prefix>/50_evaluation/...  (untagged)

Usage (from src/):
    python -m evaluation.aging_matrix /path/to/battery_config.json
    python -m evaluation.aging_matrix /path/to/battery_config.json -o /tmp/out_dir
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

from evaluation.export_cap_pulse import EVAL_DIRNAME, build_capacity_table
from util import io_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CSV_NAME = "aging_matrix.csv"
HTML_NAME = "aging_matrix.html"

GROUP_KEYS = ["C_Rate", "Temperature", "DOD", "SOC"]

# Colors for the multi-temperature surface; temperatures outside this map fall
# back to the plotly default palette.
TEMP_COLORS = {15: "skyblue", 25: "seagreen", 35: "goldenrod", 45: "tomato"}
_FALLBACK_COLORS = ["mediumpurple", "darkorange", "teal", "crimson", "slategray"]

# Pale fill for the translucent mesh shell, one step lighter than the line
# color so the wireframe stays readable on top of it. 35 °C had no notebook
# entry; khaki pairs with the goldenrod markers.
TEMP_FILL_COLORS = {
    15: "lightblue", 25: "lightgreen", 35: "khaki", 45: "lightcoral",
}
MESH_OPACITY = 0.3

# Capacity-fade plot: DOD facets, SOC colors at 25 °C, flat pastels elsewhere.
DOD_SUBPLOT_POSITIONS = {20: (1, 1), 40: (1, 2), 60: (2, 1), 80: (2, 2), 100: (3, 1)}

# RWTH corporate palette (100 % shades), inlined from the notebook's
# `rwth_colors` module (not vendored in this repo). Verify against the real
# module if the shades matter.
RWTH_SOC_COLORS = {
    10: "#00549F",  # blue
    30: "#0098A1",  # turqoise
    50: "#F6A800",  # orange
    70: "#CC071E",  # red
    90: "#A11035",  # darkred
}
# The notebook gives every non-25 °C temperature one flat pastel.
FADE_TEMP_COLORS = {15: "lavender", 35: "lavenderblush", 45: "mistyrose"}
FADE_HIGHLIGHT_TEMP = 25  # the temperature that gets per-SOC coloring
_DEFAULT_SOC_COLOR = "#000000"
_DEFAULT_TEMP_COLOR = "#7f7f7f"

# Aging C-rate -> marker symbol, as in the notebook. Keys cover both the
# numeric and the string form (`C_Rate` arrives as "05C"/"1C" from
# `add_information_METABATT`, so the notebook's numeric test never matched).
CRATE_SYMBOLS = {
    "0.5": "circle", "05c": "circle", "0.5c": "circle", "c/2": "circle",
    "1.0": "square", "1": "square", "1c": "square",
}
_DEFAULT_CRATE_SYMBOL = "diamond"


def _crate_symbol(c_rate):
    return CRATE_SYMBOLS.get(str(c_rate).strip().lower(), _DEFAULT_CRATE_SYMBOL)


def build_cell_table(df_all):
    """Reduce the per-check-up fleet table to one row per cell.

    Adds `capacity_lost` and `Delta_Ah_throughput` (max - min over the cell's
    check-ups) alongside the cell's aging conditions.
    """
    agg = df_all.groupby("Name").agg(
        DOD=("DOD", "first"),
        SOC=("SOC", "first"),
        C_Rate=("C_Rate", "first"),
        Temperature=("Temperature", "first"),
        n_CU=("Capacity_py", "count"),
        cap_max=("Capacity_py", "max"),
        cap_min=("Capacity_py", "min"),
        ah_max=("Ah_throughput", "max"),
        ah_min=("Ah_throughput", "min"),
    ).reset_index()

    agg["capacity_lost"] = agg["cap_max"] - agg["cap_min"]
    agg["Delta_Ah_throughput"] = agg["ah_max"] - agg["ah_min"]

    single = agg[agg["n_CU"] < 2]
    if not single.empty:
        logging.info(
            f"{len(single)} cell(s) have <2 check-ups (zero capacity loss): "
            + ", ".join(single["Name"].astype(str))
        )
    return agg.drop(columns=["cap_max", "cap_min", "ah_max", "ah_min"])


def build_matrix(df_cells):
    """Aggregate per-cell losses into the (C_Rate, Temperature, DOD, SOC) matrix."""
    grp = df_cells.groupby(GROUP_KEYS, dropna=False)
    matrix = grp.agg(
        candidate_count=("Name", "size"),
        cells=("Name", lambda s: "; ".join(map(str, s))),
        capacity_lost_mean=("capacity_lost", "mean"),
        capacity_lost_std=("capacity_lost", "std"),
        Delta_Ah_throughput_mean=("Delta_Ah_throughput", "mean"),
        Delta_Ah_throughput_std=("Delta_Ah_throughput", "std"),
    ).reset_index()

    matrix["capacity_lost_norm"] = (
        matrix["capacity_lost_mean"] / matrix["Delta_Ah_throughput_mean"]
    ).replace([np.inf, -np.inf], np.nan)
    return matrix.sort_values(GROUP_KEYS).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _capacity_fade_figure(df_all, c_nom):
    """3x2 DOD-faceted capacity-vs-EFC fade plot across the fleet.

    Port of the notebook `plot_plotly` / `add_temperature_data`: per cell,
    normalized capacity (``Capacity_py / Nom_Capacity``) over equivalent full
    cycles (``Ah_throughput / Nom_Capacity``), faceted by DOD, one
    lines+markers trace per (cell, temperature, SOC, C-rate) sorted by EFC.
    Colour follows the notebook: the 25 °C series are coloured per **SOC**
    from the RWTH palette, every other temperature gets one flat pastel.
    Marker symbol encodes the aging C-rate. All five facets share the same
    x/y range so they stay comparable by eye.

    Self-contained — no ``rwth_colors`` import, no ``fig.show()``, no
    hardcoded output path.
    """
    df = df_all.copy()
    df["Name_prefix"] = df["Name"].astype(str).str.split("-").str[0]
    df["cap_norm"] = df["Capacity_py"] / c_nom
    df["EFC"] = df["Ah_throughput"] / c_nom
    if "Time" not in df.columns:
        df["Time"] = pd.NaT

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=["DOD = 20%", "DOD = 40%", "DOD = 60%",
                        "DOD = 80%", "DOD = 100%", ""],
        vertical_spacing=0.13, horizontal_spacing=0.08,
        specs=[[{}, {}], [{}, {}], [{}, None]],
    )

    seen_legend = set()

    for dod, (row, col) in DOD_SUBPLOT_POSITIONS.items():
        sub = df[df["DOD"] == dod]
        for (name, temp, soc, crate), g in sub.groupby(
            ["Name_prefix", "Temperature", "SOC", "C_Rate"], dropna=False
        ):
            g = g.sort_values("EFC")
            if g.empty:
                continue
            if temp == FADE_HIGHLIGHT_TEMP:
                color = RWTH_SOC_COLORS.get(soc, _DEFAULT_SOC_COLOR)
                legend_key = (temp, soc)
                legend_name = f"T={temp}°C, SOC={soc}"
            else:
                color = FADE_TEMP_COLORS.get(temp, _DEFAULT_TEMP_COLOR)
                legend_key = (temp, None)
                legend_name = f"T={temp}°C"
            show = legend_key not in seen_legend
            seen_legend.add(legend_key)
            fig.add_trace(
                go.Scatter(
                    x=g["EFC"], y=g["cap_norm"],
                    mode="lines+markers",
                    name=legend_name,
                    legendgroup=str(legend_key),
                    showlegend=show,
                    line=dict(color=color),
                    marker=dict(color=color, size=8, symbol=_crate_symbol(crate)),
                    customdata=g[["Name_prefix", "SOC", "C_Rate", "Time"]].to_numpy(),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"Temp: {temp}°C<br>"
                        "SOC: %{customdata[1]}%<br>"
                        f"DOD: {dod}%<br>"
                        "C-Rate: %{customdata[2]}<br>"
                        "EFC: %{x:.1f}<br>Capacity: %{y:.3f}<br>"
                        "Time: %{customdata[3]}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

    x_range, y_range = _fade_axis_ranges(df)
    for row, col in DOD_SUBPLOT_POSITIONS.values():
        fig.update_xaxes(title_text="Equivalent Full Cycle", row=row, col=col,
                         showgrid=True, gridwidth=1, gridcolor="lightgrey",
                         range=x_range)
        fig.update_yaxes(title_text="Capacity (norm.)", row=row, col=col,
                         showgrid=True, gridwidth=1, gridcolor="lightgrey",
                         range=y_range)

    fig.update_layout(
        title="Battery capacity vs equivalent full cycle, by DOD",
        legend_title="SOC/C_Rate (Temperature Color Coding)",
        width=1500, height=1050,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def _fade_axis_ranges(df):
    """Shared x/y ranges over every plotted DOD facet, as in the notebook.

    Returns ``(None, None)`` when nothing falls into the DOD facets, which
    leaves plotly on autoscale rather than pinning an empty range.
    """
    plotted = df[df["DOD"].isin(DOD_SUBPLOT_POSITIONS)]
    x = plotted["EFC"].dropna()
    y = plotted["cap_norm"].dropna()
    if x.empty or y.empty:
        return None, None
    return [0, x.max() * 1.01], [y.min() * 0.99, y.max() * 1.01]


def _variance_figure(sub, title):
    """2D SOC x DOD scatter colored by cell-to-cell capacity-loss spread."""
    fig = go.Figure(go.Scatter(
        x=sub["DOD"],
        y=sub["SOC"],
        mode="markers",
        marker=dict(
            size=16,
            color=sub["capacity_lost_std"],
            colorscale="RdBu",
            opacity=0.85,
            colorbar=dict(title="capacity_lost_std"),
            line=dict(width=1, color="black"),
        ),
        customdata=sub[["candidate_count"]].to_numpy(),
        hovertemplate=(
            "DOD: %{x}%<br>SOC: %{y}%<br>"
            "loss std: %{marker.color:.4f}<br>n cells: %{customdata[0]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Depth of Discharge (DOD) %",
        yaxis_title="State of Charge (SOC) %",
        width=820,
        height=600,
    )
    return fig


def _neighbor_lines(sub, color):
    """Wireframe line traces connecting each point to its nearest neighbors."""
    traces = []
    pts = sub[["SOC", "DOD"]].to_numpy(dtype=float)
    if len(pts) < 2:
        return traces
    tree = cKDTree(pts)
    k = min(4, len(pts) - 1)
    seen = set()
    for i in range(len(pts)):
        _, idx = tree.query(pts[i], k=k + 1)
        for j in np.atleast_1d(idx)[1:]:
            edge = (min(i, j), max(i, j))
            if edge in seen:
                continue
            seen.add(edge)
            traces.append(go.Scatter3d(
                x=[sub.iloc[i]["SOC"], sub.iloc[j]["SOC"]],
                y=[sub.iloc[i]["DOD"], sub.iloc[j]["DOD"]],
                z=[sub.iloc[i]["capacity_lost_norm"], sub.iloc[j]["capacity_lost_norm"]],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False,
                hoverinfo="skip",
            ))
    return traces


def _surface_figure(sub, title):
    """Single-group 3D aging surface: normalized loss over the SOC/DOD plane."""
    fig = go.Figure()
    for tr in _neighbor_lines(sub, "green"):
        fig.add_trace(tr)
    fig.add_trace(go.Scatter3d(
        x=sub["SOC"],
        y=sub["DOD"],
        z=sub["capacity_lost_norm"],
        mode="markers",
        marker=dict(
            size=11,
            color=sub["capacity_lost_norm"],
            colorscale="Viridis",
            opacity=1.0,
            colorbar=dict(title="norm. loss"),
            line=dict(width=2, color="black"),
        ),
        hovertemplate=(
            "SOC: %{x}%<br>DOD: %{y}%<br>norm. loss: %{z:.4f}<extra></extra>"
        ),
        name="cells",
    ))
    _layout_3d(fig, title)
    return fig


def _mesh_surface(sub, color, name):
    """Translucent shell over the SOC/DOD plane for one temperature.

    Plotly triangulates internally: with no `i/j/k` given, `Mesh3d` defaults
    to `alphahull=-1`, i.e. a Delaunay triangulation in the plane named by
    `delaunayaxis` ("z" — our SOC/DOD plane). One trace per temperature, so
    no `scipy.spatial.Delaunay` and none of the notebook's per-triangle
    traces. Returns None when there are too few points to triangulate.
    """
    if len(sub) < 3:
        return None
    return go.Mesh3d(
        x=sub["SOC"],
        y=sub["DOD"],
        z=sub["capacity_lost_norm"],
        delaunayaxis="z",
        color=color,
        opacity=MESH_OPACITY,
        showscale=False,
        showlegend=False,
        hoverinfo="skip",
        name=f"surface {name}",
    )


def _multi_temp_figure(df_crate, c_rate):
    """Overlay the 3D aging surface for every temperature at one C-rate."""
    fig = go.Figure()
    temps = sorted(df_crate["Temperature"].dropna().unique())
    for n, temp in enumerate(temps):
        sub = df_crate[df_crate["Temperature"] == temp]
        if sub.empty:
            continue
        color = TEMP_COLORS.get(temp, _FALLBACK_COLORS[n % len(_FALLBACK_COLORS)])
        mesh = _mesh_surface(sub, TEMP_FILL_COLORS.get(temp, color), f"{temp}°C")
        if mesh is not None:
            fig.add_trace(mesh)
        for tr in _neighbor_lines(sub, color):
            fig.add_trace(tr)
        fig.add_trace(go.Scatter3d(
            x=sub["SOC"],
            y=sub["DOD"],
            z=sub["capacity_lost_norm"],
            mode="markers",
            marker=dict(size=11, color=color, opacity=1.0,
                        line=dict(width=2, color="black")),
            hovertemplate=(
                f"{temp}°C<br>"
                "SOC: %{x}%<br>DOD: %{y}%<br>norm. loss: %{z:.4f}<extra></extra>"
            ),
            name=f"{temp}°C",
        ))
    _layout_3d(fig, f"Aging surface — C-Rate {c_rate}, all temperatures")
    fig.update_layout(showlegend=True)
    return fig


def _layout_3d(fig, title):
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="SOC %",
            yaxis_title="DOD %",
            zaxis_title="Norm. capacity loss / Ah throughput",
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.2)),
        ),
        width=900,
        height=700,
    )


def render_html(matrix, df_all=None, c_nom=None, title="Battery aging matrix"):
    """Assemble all plots into one self-contained HTML string."""
    sections = []  # (heading, figure)
    if df_all is not None and not df_all.empty:
        sections.append(
            ("Capacity fade — capacity vs equivalent full cycle",
             _capacity_fade_figure(df_all, c_nom))
        )
    for (c_rate, temp), sub in matrix.groupby(["C_Rate", "Temperature"], dropna=False):
        if sub.empty:
            continue
        label = f"C-Rate {c_rate}, {temp}°C"
        sections.append((f"Variance — {label}", _variance_figure(sub, f"Cell-to-cell variance — {label}")))
        sections.append((f"Surface — {label}", _surface_figure(sub, f"Aging surface — {label}")))

    for c_rate, df_crate in matrix.groupby("C_Rate", dropna=False):
        if df_crate["Temperature"].nunique(dropna=True) > 1:
            sections.append(
                (f"Multi-temperature surface — C-Rate {c_rate}",
                 _multi_temp_figure(df_crate, c_rate))
            )

    parts = []
    for i, (heading, fig) in enumerate(sections):
        parts.append(f"<h2>{heading}</h2>")
        parts.append(fig.to_html(
            full_html=False, include_plotlyjs="cdn" if i == 0 else False
        ))
    body = "".join(parts) if parts else "<p>No aging-matrix data to plot.</p>"
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body>"
        f"<h1>{title}</h1>{body}</body></html>"
    )


# --------------------------------------------------------------------------- #
# Output routing
# --------------------------------------------------------------------------- #

def _write_outputs(matrix, cfg, out_dir, df_all=None):
    html = render_html(matrix, df_all=df_all, c_nom=cfg["nom_capacity"])

    if io_router.writes_local(cfg) and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, CSV_NAME)
        html_path = os.path.join(out_dir, HTML_NAME)
        matrix.to_csv(csv_path, index=False)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        logging.info(f"Wrote {csv_path}")
        logging.info(f"Wrote {html_path}")

    if io_router.writes_minio(cfg):
        client = io_router.make_minio_client(cfg)
        io_router.upload_csv(
            client, cfg, matrix, f"{EVAL_DIRNAME}/{CSV_NAME}", include_tag=False
        )
        io_router._upload_bytes(
            client, cfg, f"{EVAL_DIRNAME}/{HTML_NAME}",
            html.encode("utf-8"), include_tag=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Build the fleet-wide aging matrix (Alterungsmatrix)"
    )
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Override local output directory (default: <working_path>/50_evaluation)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    source = cfg.get("download_from", "local")
    df_all = build_capacity_table(cfg, source=source)
    if df_all.empty:
        logging.warning("No capacity data found")
        return

    if "Ah_throughput" not in df_all.columns or df_all["Ah_throughput"].isna().all():
        logging.error(
            "capacity CSVs carry no Ah_throughput — re-run the pipeline so "
            "export_capacity writes it, then retry"
        )
        return

    df_cells = build_cell_table(df_all)
    matrix = build_matrix(df_cells)
    logging.info(
        f"Aging matrix: {len(matrix)} (C_Rate, Temperature, DOD, SOC) cells "
        f"from {len(df_cells)} cells"
    )

    out_dir = args.output_dir or os.path.join(
        cfg.get("working_path", "."), EVAL_DIRNAME
    )
    _write_outputs(matrix, cfg, out_dir, df_all=df_all)


if __name__ == "__main__":
    main()
