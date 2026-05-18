"""Aging-status monitor: build a sortable HTML table of cell SOH.

Reads per-cell capacity CSVs from `40_capacity_monitore/` (local or MinIO,
chosen by cfg["download_from"]).

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
import pyarrow.parquet as pq

from util import io_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DEFAULT_RUNNING_WINDOW_DAYS = 2
YELLOW_THRESHOLD = 70.0
RED_THRESHOLD = 60.0


def _make_readers(cfg, source):
    """Return (cells, fetch_capacity, fetch_gold_tail).

    fetch_gold_tail(stem) returns a 1-row DataFrame with Time + Prozedur of the
    last row in the cell's GOLD parquet, or None if GOLD is missing.
    """
    if source == "local":
        wp = cfg["working_path"]
        cap_dir = os.path.join(wp, "40_capacity_monitore")
        files = sorted(glob.glob(os.path.join(cap_dir, "*_capacity.csv")))
        cells = [os.path.basename(p) for p in files]

        def fetch_capacity(name):
            return pd.read_csv(os.path.join(cap_dir, name))

        def fetch_gold_tail(stem):
            path = os.path.join(wp, "GOLD", f"{stem}.parquet")
            if not os.path.exists(path):
                return None
            return _read_gold_tail(path)

        return cells, fetch_capacity, fetch_gold_tail

    client = io_router.make_minio_client(cfg)
    bucket = cfg["bucket_name"]
    base = f"{cfg['minio_prefix']}/40_capacity_monitore/"
    objs = client.list_objects(bucket, prefix=base, recursive=False)
    cells = sorted(
        os.path.basename(o.object_name)
        for o in objs
        if o.object_name.endswith("_capacity.csv")
    )

    def fetch_capacity(name):
        key = f"{base}{name}"
        response = client.get_object(bucket, key)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        return pd.read_csv(io.BytesIO(data))

    def fetch_gold_tail(stem):
        try:
            f = io_router.open_gold_range(client, cfg, f"{stem}.parquet")
        except Exception:
            return None
        try:
            return _read_gold_tail(f)
        finally:
            f.close()

    return cells, fetch_capacity, fetch_gold_tail


def _read_gold_tail(source):
    """Read Time + Prozedur of the last row from a parquet path or BytesIO,
    using only the last row group to avoid loading the full file."""
    pf = pq.ParquetFile(source)
    available = pf.schema_arrow.names
    cols = [c for c in ("Time", "Prozedur") if c in available]
    if not cols or pf.num_row_groups == 0:
        return None
    last_rg = pf.read_row_group(pf.num_row_groups - 1, columns=cols).to_pandas()
    return last_rg.tail(1) if not last_rg.empty else None


def _is_unfinished(cell_stem):
    """True if the BRONZE_CU stem ends with `=unfinished` (download status tag)."""
    return cell_stem.endswith("=unfinished")


def _cell_summary(df):
    if df.empty or "SOH" not in df.columns:
        return {"latest_soh": None, "delta_soh_per_cu": None, "n_cu": 0}

    df = df.sort_values("BM_Programm")
    soh_series = pd.to_numeric(df["SOH"], errors="coerce").dropna()
    n_cu = int(len(soh_series))
    latest = float(soh_series.iloc[-1]) if n_cu else None

    if n_cu >= 2:
        tail = soh_series.tail(5).reset_index(drop=True)
        x = pd.Series(range(len(tail)), dtype=float)
        slope = float(((x - x.mean()) * (tail - tail.mean())).sum() / ((x - x.mean()) ** 2).sum())
    else:
        slope = None

    return {"latest_soh": latest, "delta_soh_per_cu": slope, "n_cu": n_cu}


def build_status_table(cfg, source="minio"):
    cells, fetch_capacity, fetch_gold_tail = _make_readers(cfg, source)
    logging.info(f"Found {len(cells)} capacity CSVs ({source})")

    running_window_days = cfg.get("running_window_days", DEFAULT_RUNNING_WINDOW_DAYS)
    now = datetime.now(timezone.utc)
    running_cutoff = now - timedelta(days=running_window_days)

    rows = []
    for cell in cells:
        cell_stem = cell.replace("_capacity.csv", "")
        try:
            df = fetch_capacity(cell)
            s = _cell_summary(df)
        except Exception as e:
            logging.warning(f"{cell}: {type(e).__name__}: {e}")
            continue

        last_t = pd.NaT
        last_prozedur = None
        try:
            tail = fetch_gold_tail(cell_stem)
            if tail is not None and not tail.empty:
                if "Time" in tail.columns:
                    last_t = pd.to_datetime(tail["Time"].iloc[0], errors="coerce")
                if "Prozedur" in tail.columns:
                    last_prozedur = tail["Prozedur"].iloc[0]
        except Exception as e:
            logging.warning(f"{cell_stem}: GOLD tail read failed: {type(e).__name__}: {e}")

        if _is_unfinished(cell_stem):
            status = "unfinished"
        else:
            if pd.notna(last_t):
                last_t_aware = last_t.tz_localize("UTC") if last_t.tzinfo is None else last_t
                is_running = last_t_aware >= running_cutoff
            else:
                is_running = False
            status = "running" if is_running else "finished"

        rows.append({
            "cell": cell_stem,
            "latest_SOH_%": s["latest_soh"],
            "dSOH_per_CU": s["delta_soh_per_cu"],
            "n_CU": s["n_cu"],
            "last_row_time": last_t,
            "last_Prozedur": last_prozedur,
            "status": status,
        })

    df_status = pd.DataFrame(rows)
    if df_status.empty:
        return df_status

    df_status["_status_order"] = df_status["status"].map(
        {"unfinished": 0, "running": 1, "finished": 2}
    )
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


def render_html(df, out_path, running_window_days=DEFAULT_RUNNING_WINDOW_DAYS):
    headers = ["cell", "latest_SOH_%", "dSOH_per_CU", "n_CU", "last_row_time", "last_Prozedur", "status"]
    df_unfinished = df[df["status"] == "unfinished"]
    df_running = df[df["status"] == "running"]
    df_finished = df[df["status"] == "finished"]

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    tables = (
        _render_table(df_unfinished, headers, "t_unfinished", "Unfinished")
        + "\n"
        + _render_table(df_running, headers, "t_running", "Running")
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
<div class="meta">Generated {generated} &middot; {len(df)} cells &middot; yellow &lt; {YELLOW_THRESHOLD:.0f}% SOH, red &lt; {RED_THRESHOLD:.0f}% SOH &middot; unfinished = BRONZE_CU stem ends with `=unfinished` &middot; running = BRONZE finished but last GOLD row within {running_window_days} days</div>
{tables}
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<script>
$(function() {{
  $('#t_unfinished').DataTable({{ paging: false, order: [], info: true, searching: true }});
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
    running_window_days = cfg.get("running_window_days", DEFAULT_RUNNING_WINDOW_DAYS)
    df = build_status_table(cfg, source=source)
    if df.empty:
        logging.warning("No cells found")
        return

    out = args.output or os.path.join(
        cfg.get("working_path", "."), "40_capacity_monitore", "aging_status.html"
    )
    if io_router.writes_local(cfg):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        render_html(df, out, running_window_days=running_window_days)
    else:
        # Need a local temp render to get the HTML payload
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w")
        tmp.close()
        render_html(df, tmp.name, running_window_days=running_window_days)
        out = tmp.name

    if io_router.writes_minio(cfg):
        client = io_router.make_minio_client(cfg)
        with open(out, "rb") as f:
            payload = f.read()
        key = f"40_capacity_monitore/aging_status.html"
        io_router._upload_bytes(client, cfg, key, payload, include_tag=False)


if __name__ == "__main__":
    main()
