#! /bin/env python3

import pandas as pd
import requests


def barcode_to_id(barcode):
    barcode = barcode.upper().strip().replace(" ", "")
    return int(barcode[2:8], 36)


def get_properties(specimen, apikey):
    sid = specimen["id"]
    rSpecimen = requests.get(
        f"https://ahjo1.isea.rwth-aachen.de/api/specimen/{sid}",
        headers={"key": apikey},
    ).json()
    rProperties = requests.get(
        f"https://ahjo1.isea.rwth-aachen.de/api/specimen/{sid}/properties",
        headers={"key": apikey},
    )
    properties = {
        "ID": specimen["id"],
        "Name": specimen["name"],
        "Species": rSpecimen["species"]["manufacturer"]
        + " "
        + rSpecimen["species"]["typename"],
        **{p["name"]: p["value"] for p in rProperties.json()},
    }
    return properties


def main(project_id, apikey, output):
    r = requests.get(
        f"https://ahjo.isea.rwth-aachen.de/api/specimen?project={project_id}",
        headers={"key": apikey},
    )
    properties = [get_properties(specimen, apikey) for specimen in r.json()]
    df = pd.DataFrame.from_records(properties)
    if output:
        df.to_csv(output, index=False)
    else:
        print(df.to_csv(index=False))
