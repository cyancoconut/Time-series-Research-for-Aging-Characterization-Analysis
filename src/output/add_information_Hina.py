"""Derive aging conditions (C_Rate, Temperature, SOC, DOD) from the HiNa aging
procedure name.

Per-project parser for the HiNa campaign. The aging procedure name is the
cycling test file's filename field (the ``Cyc`` procedure, as opposed to the
``RPT`` check-up). It is parsed by fixed field position after splitting on
``_``:

    csi  HiNa  Cyc   2C     45    mean50dod100   15
     0    1     2     3      4         5           6
                    C_Rate  Temp    SOC+DOD     cell number

    parts[3] -> C_Rate       (token string, e.g. "2C")
    parts[4] -> Temperature  (int, °C)
    parts[5] -> "mean<SOC>dod<DOD>"  -> SOC = 50, DOD = 100
    parts[6] -> cell number  (ignored)

Mirrors the ``add_information_METABATT`` interface
(``add_additional_information(df_results)``) so it can be swapped in for the
HiNa campaign.
"""

import pandas as pd

# Aging procedures carry this marker; the check-up (``RPT``) does not.
AGING_MARKER = "Cyc"


def parse_aging_name(name):
    """Parse one HiNa aging procedure name into its conditions by field position.

    Returns a dict with any of ``C_Rate`` (str), ``Temperature`` (int),
    ``SOC`` (int), ``DOD`` (int) that could be read; missing/malformed fields
    are omitted. Returns ``{}`` for an empty / non-string name.
    """
    out = {}
    if not isinstance(name, str) or not name:
        return out

    parts = name.split("_")

    # C_Rate (parts[3]) — kept as the token string, e.g. "2C".
    if len(parts) > 3:
        out["C_Rate"] = parts[3]

    # Temperature (parts[4]) — bare integer in °C.
    if len(parts) > 4 and parts[4].isdigit():
        out["Temperature"] = int(parts[4])

    # SOC / DOD (parts[5]) — "mean<SOC>dod<DOD>".
    if len(parts) > 5:
        soc, dod = _split_soc_dod(parts[5])
        if soc is not None:
            out["SOC"] = soc
        if dod is not None:
            out["DOD"] = dod

    return out


def _split_soc_dod(field):
    """Split a ``mean<SOC>dod<DOD>`` field into (SOC, DOD) ints, or (None, None)."""
    if "dod" not in field:
        return None, None
    soc_part, dod_part = field.split("dod", 1)
    soc_part = soc_part.replace("mean", "")
    soc = int(soc_part) if soc_part.isdigit() else None
    dod = int(dod_part) if dod_part.isdigit() else None
    return soc, dod


def select_aging_name(procedures):
    """Pick the aging (cycling) procedure name from a cell's procedure list.

    ``procedures`` may be a list of names or a single string. The aging
    condition is constant per cell, so there is normally one cycling name;
    the ``RPT`` check-up is filtered out by the ``Cyc`` marker.
    """
    if isinstance(procedures, str):
        procedures = [procedures]
    for name in procedures or []:
        if isinstance(name, str) and AGING_MARKER in name:
            return name
    return None


def add_additional_information(df_results):
    """Add DOD/SOC/C_Rate/Temperature columns from the ``Procedures`` column.

    The condition is constant per cell, so every row of a cell gets the same
    values. Unresolved fields are left as NA.
    """
    conditions = df_results["Procedures"].apply(
        lambda procs: parse_aging_name(select_aging_name(procs))
    )
    df_results["DOD"] = conditions.apply(lambda c: c.get("DOD", pd.NA))
    df_results["SOC"] = conditions.apply(lambda c: c.get("SOC", pd.NA))
    df_results["C_Rate"] = conditions.apply(lambda c: c.get("C_Rate", pd.NA))
    df_results["Temperature"] = conditions.apply(lambda c: c.get("Temperature", pd.NA))
