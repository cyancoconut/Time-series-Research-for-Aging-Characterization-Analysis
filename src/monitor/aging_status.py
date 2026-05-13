"""Aging-status monitor: build a sortable HTML table of cell SOH from MinIO GOLD.

Usage (from src/):
    python -m monitor.aging_status /path/to/battery_config.json
    python -m monitor.aging_status /path/to/battery_config.json -o out.html
"""

import argparse
import glob
import io
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from util import io_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

RUNNING_WINDOW_DAYS = 14
YELLOW_THRESHOLD = 70.0
RED_THRESHOLD = 60.0


def _make_reader(cfg, source):
    if source == "local":
        wp = cfg["working_path"]
        return (
            io_router.list_gold_cells_local(wp),
            lambda cell: pd.read_parquet(io_router.gold_local_path(wp, cell)),
        )
    client = io_router.make_minio_client(cfg)
    return (
        io_router.list_gold_cells(client, cfg),
        lambda cell: pd.read_parquet(io.BytesIO(io_router.fetch_gold_bytes(client, cfg, cell))),
    )


def _cell_summary(df, nom_capacity):
    if "Time" in df.columns and not df["Time"].empty:
        times = pd.to_datetime(df["Time"], errors="coerce")
        last_row_time = times.max()
        if "Prozedur" in df.columns and pd.notna(last_row_time):
            last_prozedur = df.loc[times.idxmax(), "Prozedur"]
        else:
            last_prozedur = None
    else:
        last_row_time = pd.NaT
        last_prozedur = None

    cap = df[df["target"] == "CAP"]
    if cap.empty or "Capacity_py" not in cap.columns:
        return {
            "latest_soh": None,
            "delta_soh_per_cu": None,
            "n_cu": 0,
            "last_row_time": last_row_time,
            "last_prozedur": last_prozedur,
        }

    cap_by_prog = (
        cap.groupby("BM_Programm")["Capacity_py"]
        .first()
        .dropna()
        .sort_index()
    )
    if cap_by_prog.empty:
        return {
            "latest_soh": None,
            "delta_soh_per_cu": None,
            "n_cu": 0,
            "last_row_time": last_row_time,
            "last_prozedur": last_prozedur,
        }

    soh_series = cap_by_prog / nom_capacity * 100.0
    latest = float(soh_series.iloc[-1])
    n_cu = int(len(soh_series))

    # Slope per CU using the last up-to-5 points
    if n_cu >= 2:
        tail = soh_series.tail(5)
        x = pd.Series(range(len(tail)), index=tail.index, dtype=float)
        slope = float(((x - x.mean()) * (tail - tail.mean())).sum() / ((x - x.mean()) ** 2).sum())
    else:
        slope = None

    return {
        "latest_soh": latest,
        "delta_soh_per_cu": slope,
        "n_cu": n_cu,
        "last_row_time": last_row_time,
        "last_prozedur": last_prozedur,
    }


def build_status_table(cfg, source="minio"):
    nom_capacity = cfg["nom_capacity"]
    cells, fetch = _make_reader(cfg, source)
    logging.info(f"Found {len(cells)} GOLD parquets ({source})")

    now = datetime.now(timezone.utc)
    running_cutoff = now - timedelta(days=RUNNING_WINDOW_DAYS)

    rows = []
    for cell in cells:
        try:
            df = fetch(cell)
            s = _cell_summary(df, nom_capacity)
        except Exception as e:
            logging.warning(f"{cell}: {type(e).__name__}: {e}")
            continue

        last_t = s["last_row_time"]
        if pd.notna(last_t):
            last_t_aware = last_t.tz_localize("UTC") if last_t.tzinfo is None else last_t
            is_running = last_t_aware >= running_cutoff
        else:
            is_running = False

        rows.append({
            "cell": cell,
            "latest_SOH_%": s["latest_soh"],
            "dSOH_per_CU": s["delta_soh_per_cu"],
            "n_CU": s["n_cu"],
            "last_row_time": last_t,
            "last_Prozedur": s["last_prozedur"],
            "status": "running" if is_running else "finished",
        })

    df_status = pd.DataFrame(rows)
    if df_status.empty:
        return df_status

    df_status["_status_order"] = df_status["status"].map({"running": 0, "finished": 1})
    df_status["_soh_sort"] = df_status["latest_SOH_%"].fillna(float("inf"))
    df_status = df_status.sort_values(
        by=["_status_order", "_soh_sort"], ascending=[True, True]
    ).drop(columns=["_status_order", "_soh_sort"]).reset_index(drop=True)
    return df_status


def _row_class(soh):
    if soh is None or pd.isna(soh):
        return ""
    if soh < RED_THRESHOLD:
        return "red"
    if soh < YELLOW_THRESHOLD:
        return "yellow"
    return ""


def _render_rows(df, headers):
    rows = []
    for _, r in df.iterrows():
        cls = _row_class(r["latest_SOH_%"])
        tds = []
        for h in headers:
            v = r[h]
            if h == "latest_SOH_%" and pd.notna(v):
                cell_html = f"{v:.1f}"
            elif h == "dSOH_per_CU" and pd.notna(v):
                cell_html = f"{v:+.2f}"
            elif h == "last_row_time" and pd.notna(v):
                cell_html = pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")
            elif pd.isna(v):
                cell_html = ""
            else:
                cell_html = str(v)
            tds.append(f"<td>{cell_html}</td>")
        rows.append(f'<tr class="{cls}">{"".join(tds)}</tr>')
    return "\n".join(rows)


def _render_table(df, headers, table_id, title):
    thead = "".join(f"<th>{h}</th>" for h in headers)
    body = _render_rows(df, headers)
    return f"""<h2>{title} ({len(df)})</h2>
<table id="{table_id}" class="display compact">
<thead><tr>{thead}</tr></thead>
<tbody>
{body}
</tbody></table>"""


def render_html(df, out_path):
    headers = ["cell", "latest_SOH_%", "dSOH_per_CU", "n_CU", "last_row_time", "last_Prozedur", "status"]
    df_running = df[df["status"] == "running"]
    df_finished = df[df["status"] == "finished"]

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    tables = (
        _render_table(df_running, headers, "t_running", "Running")
        + "\n"
        + _render_table(df_finished, headers, "t_finished", "Finished")
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Cell aging status</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; margin: 1.5rem; }}
 h1 {{ font-size: 1.2rem; }}
 h2 {{ font-size: 1.05rem; margin-top: 1.5rem; }}
 table.dataTable tbody tr.yellow {{ background: #fff3b0 !important; }}
 table.dataTable tbody tr.red    {{ background: #f4a8a8 !important; }}
 table.dataTable tbody tr.yellow:hover td,
 table.dataTable tbody tr.red:hover td {{ background: inherit !important; }}
 .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1rem; }}
</style>
</head><body>
<h1>Cell aging status</h1>
<div class="meta">Generated {generated} &middot; {len(df)} cells &middot; yellow &lt; {YELLOW_THRESHOLD:.0f}% SOH, red &lt; {RED_THRESHOLD:.0f}% SOH &middot; running = last row within {RUNNING_WINDOW_DAYS} days</div>
{tables}
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<script>
$(function() {{
  $('#t_running').DataTable({{ paging: false, order: [], info: true, searching: true }});
  $('#t_finished').DataTable({{ paging: false, order: [], info: true, searching: true }});
}});
</script>
</body></html>"""
    with open(out_path, "w") as f:
        f.write(html)
    logging.info(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cell aging-status report")
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument("-o", "--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    source = cfg.get("download_from", "local")
    df = build_status_table(cfg, source=source)
    if df.empty:
        logging.warning("No cells found")
        return

    out = args.output or os.path.join(
        cfg.get("working_path", "."), "40_capacity_monitore", "aging_status.html"
    )
    if io_router.writes_local(cfg):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        render_html(df, out)
    else:
        # Need a local temp render to get the HTML payload
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w")
        tmp.close()
        render_html(df, tmp.name)
        out = tmp.name

    if io_router.writes_minio(cfg):
        client = io_router.make_minio_client(cfg)
        with open(out, "rb") as f:
            payload = f.read()
        key = f"40_capacity_monitore/aging_status.html"
        io_router._upload_bytes(client, cfg, key, payload, include_tag=False)


if __name__ == "__main__":
    main()
