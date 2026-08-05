"""Load EIS (electrochemical impedance spectroscopy) measurement files.

EIS measurements are produced by a dedicated device (channel ``EISkanal``),
separate from the cycler (``Kreis``). Each measurement is one file — CSV or
parquet — whose rows are the per-second dwell samples of a frequency sweep.
The device holds each frequency for several seconds while it settles, so the
raw file has many rows per frequency with the impedance columns filling in only
once the point has settled. This module reduces that to a clean one-point-per-
frequency spectrum.

A file is recognised as an EIS measurement by its filename: the cycler encodes
metadata as ``=``-delimited fields, and an EIS device file carries a
measurement token matching ``EIS\\d+`` or ``INS\\d+`` (e.g. ``EIS00017``,
``INS00003``) where an ordinary cycler test carries ``TS\\d+``. That token is
the marker (configurable) and doubles as the measurement number.

Filename layouts differ between the raw CSV export and the pipeline-downloaded
parquet, but both share:
  - the **cell stem** at ``=``-field index 1
  - a **datetime** field (test start), ``YYYY-MM-DD[ _]HHMMSS``
  - the **EIS\\d+** measurement token

so metadata is parsed layout-agnostically by scanning the fields rather than
hardcoding positions.

Canonical spectrum columns returned by :func:`load_eis_spectrum`:
    Time, frequency, Z_real, Z_imag, Z_abs, phase, U
"""

import glob
import io
import os
import re

import pandas as pd

# Raw EIS columns -> canonical names.
_EIS_RENAME = {
    "Zeit": "Time",
    "ActFreq": "frequency",
    "Zreal1": "Z_real",
    "Zimg1": "Z_imag",
    "Betrag": "Z_abs",
    "Phase": "phase",
    "U1": "U",
}
_CANONICAL_COLS = ["Time", "frequency", "Z_real", "Z_imag", "Z_abs", "phase", "U"]
# Columns that must be present for a file to count as a real EIS measurement.
_REQUIRED_RAW = ["ActFreq", "Zreal1", "Zimg1", "Betrag"]

# Default marker: the measurement token in an =-delimited field. EIS-device
# files carry "EIS<n>" or "INS<n>" (e.g. EIS00017, INS00003) where an ordinary
# cycler test carries "TS<n>".
DEFAULT_EIS_FILE_MARKER = r"(?:EIS|INS)\d+"
# Datetime token used as the test-date field (handles "_" or " " separators).
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ _]\d{6}")


def _fields(filename: str) -> list:
    """The ``=``-delimited fields of a basename (extension stripped)."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split("=")


def is_eis_file(filename: str, marker: str = DEFAULT_EIS_FILE_MARKER) -> bool:
    """True when any ``=``-field carries the EIS measurement token.

    The token (``EIS\\d+`` by default) is present only on EIS-device files;
    ordinary cycler tests carry ``TS\\d+`` and EIS-containing *procedure* names
    (``zho_Namey_EIS``, ``..._EIS_CHA``) have no trailing digits, so they do not
    match.
    """
    pat = re.compile(marker)
    return any(pat.search(f) for f in _fields(filename))


def parse_eis_filename(filename: str, marker: str = DEFAULT_EIS_FILE_MARKER) -> dict:
    """Extract ``cell_stem``, ``test_date`` and ``eis_number`` from the name.

    Layout-agnostic: cell stem is field 1; the date is the datetime-looking
    field; the EIS number is the ``EIS\\d+`` token. Missing pieces come back as
    ``None`` (``test_date`` as ``NaT``).
    """
    fields = _fields(filename)
    cell_stem = fields[1] if len(fields) > 1 else None

    eis_number = None
    tok = re.compile(marker)
    for f in fields:
        m = tok.search(f)
        if m:
            eis_number = m.group(0)
            break

    test_date = pd.NaT
    for f in fields:
        m = _DATE_RE.search(f)
        if m:
            test_date = pd.to_datetime(m.group(0).replace("_", " "), errors="coerce")
            break

    return {"cell_stem": cell_stem, "test_date": test_date, "eis_number": eis_number}


def _read_raw(source) -> pd.DataFrame:
    """Read an EIS file (path/bytes) into a raw DataFrame.

    CSV files carry a units sub-header on line 2 (``skiprows=[1]``); parquet
    files do not. ``source`` is a filesystem path or a ``bytes`` payload; for
    bytes the format is sniffed from the parquet magic.
    """
    if isinstance(source, (bytes, bytearray)):
        buf = io.BytesIO(source)
        if source[:4] == b"PAR1":
            return pd.read_parquet(buf)
        return pd.read_csv(buf, skiprows=[1])

    if str(source).lower().endswith(".parquet"):
        return pd.read_parquet(source)
    return pd.read_csv(source, skiprows=[1])


def reduce_spectrum(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse the dwell-sampled sweep to one settled point per frequency.

    Keeps rows that carry a real measurement (``ActFreq > 0`` and ``Betrag > 0``
    — the pre-settling rows log zeros), then takes the **last** sample per
    ``ActFreq`` (the settled value) and sorts ascending by frequency. Returns
    the canonical-column spectrum.
    """
    valid = raw[(raw["ActFreq"] > 0) & (raw["Betrag"] > 0)]
    if valid.empty:
        return pd.DataFrame(columns=_CANONICAL_COLS)
    settled = valid.groupby("ActFreq", sort=False).tail(1)
    spec = settled.rename(columns=_EIS_RENAME)
    spec = spec[[c for c in _CANONICAL_COLS if c in spec.columns]].copy()
    spec["Time"] = pd.to_datetime(spec["Time"], errors="coerce")
    return spec.sort_values("frequency").reset_index(drop=True)


def load_eis_spectrum(source, filename: str = None,
                      marker: str = DEFAULT_EIS_FILE_MARKER):
    """Load one EIS file and return ``(spectrum_df, meta)`` or ``(None, meta)``.

    ``source`` is a path or bytes; ``filename`` overrides the name used for
    metadata (needed when ``source`` is bytes). Returns ``(None, meta)`` when the
    file lacks the EIS impedance columns (mis-named / not an actual spectrum) or
    yields no settled points, so callers can skip it with a warning.

    ``meta`` carries ``cell_stem``, ``test_date``, ``eis_number``, ``meas_time``
    (first settled timestamp), ``n_points`` and ``mean_U``.
    """
    name = filename or (source if isinstance(source, str) else "")
    meta = parse_eis_filename(name, marker)
    meta.update({"meas_time": pd.NaT, "n_points": 0, "mean_U": float("nan")})

    raw = _read_raw(source)
    if not all(c in raw.columns for c in _REQUIRED_RAW):
        return None, meta

    spec = reduce_spectrum(raw)
    if spec.empty:
        return None, meta

    # Stamp measurement identity onto every spectrum row for the bundled export.
    spec["eis_number"] = meta["eis_number"]
    spec["test_date"] = meta["test_date"]
    meta["meas_time"] = spec["Time"].min()
    meta["n_points"] = len(spec)
    meta["mean_U"] = float(spec["U"].mean()) if "U" in spec else float("nan")
    return spec, meta


def list_eis_files_local(root: str, cell_stem: str,
                         marker: str = DEFAULT_EIS_FILE_MARKER) -> list:
    """EIS file paths in ``<root>/<cell_stem>/`` (csv + parquet, marker-filtered).

    Mirrors the per-test local layout used by ``build_bronze_cu_with_ah`` — a
    cell's raw files live in its own subfolder next to the cycler parquets.
    """
    cell_dir = os.path.join(root, cell_stem)
    if not os.path.isdir(cell_dir):
        return []
    paths = glob.glob(os.path.join(cell_dir, "*.csv")) + glob.glob(
        os.path.join(cell_dir, "*.parquet")
    )
    return sorted(p for p in paths if is_eis_file(p, marker))


def _to_naive(series: pd.Series) -> pd.Series:
    """Datetime series as tz-naive wall-clock (drops any tz).

    BRONZE_CU ``Zeit`` is tz-aware UTC while the EIS files log naive timestamps;
    the EIS device shares the cycler's lab clock (UTC), so comparing on the
    UTC wall-clock aligns them. tz-naive input passes through unchanged.
    """
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s


def list_eis_files_minio(client, cfg: dict, cell_stem: str,
                         marker: str = DEFAULT_EIS_FILE_MARKER) -> list:
    """EIS object names under ``<minio_prefix>/<cell_stem>/`` (csv + parquet).

    Mirrors the per-test MinIO layout (a cell's raw files live under its own
    prefix). Returns full object names, marker-filtered.
    """
    bucket = cfg["bucket_name"]
    base = f"{cfg['minio_prefix']}/{cell_stem}/"
    objs = client.list_objects(bucket, prefix=base, recursive=True)
    return sorted(
        o.object_name
        for o in objs
        if o.object_name.endswith((".csv", ".parquet"))
        and is_eis_file(o.object_name, marker)
    )


def fetch_eis_bytes(client, cfg: dict, object_name: str) -> bytes:
    """Fetch one EIS object's raw payload from MinIO."""
    response = client.get_object(cfg["bucket_name"], object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def match_spectra_to_programs(metas, anchors, tol_minutes: float):
    """Map each measurement to a ``BM_Programm`` by nearest anchor time.

    ``metas`` is a list of per-measurement dicts (from :func:`load_eis_spectrum`)
    carrying ``meas_time``. ``anchors`` is a DataFrame with ``BM_Programm`` and a
    ``Time`` column describing the cell's EIS-labeled segments. Each measurement
    is assigned the ``BM_Programm`` of the nearest anchor within
    ``tol_minutes``; unmatched measurements map to ``None``.

    Returns a list of ``BM_Programm`` (or ``None``) aligned with ``metas``.
    """
    if anchors is None or anchors.empty:
        return [None] * len(metas)
    a = anchors.copy()
    a["Time"] = _to_naive(a["Time"])
    a = a.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    if a.empty:
        return [None] * len(metas)

    tol = pd.Timedelta(minutes=tol_minutes)
    out = []
    for m in metas:
        t = _to_naive(pd.Series([m.get("meas_time")])).iloc[0]
        if pd.isna(t):
            out.append(None)
            continue
        delta = (a["Time"] - t).abs()
        i = int(delta.values.argmin())
        out.append(a.iloc[i]["BM_Programm"] if delta.iloc[i] <= tol else None)
    return out
