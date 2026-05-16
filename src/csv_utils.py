# -*- coding: utf-8 -*-
"""
csv_utils.py
============
Shared utility for exporting the data behind every plot to CSV.

Directory convention
--------------------
Plot path   : outputs/plots/<subdir>/<figure>.png
CSV path    : outputs/csv/<subdir>/<figure>[__<suffix>].csv

Usage
-----
from csv_utils import csv_save, plot_to_csv

# One table per figure:
csv_save({"col_a": arr_a, "col_b": arr_b}, out_png_path)

# Multiple tables for subplots (use a suffix per panel):
csv_save(df_panel1, out_png_path, suffix="__train")
csv_save(df_panel2, out_png_path, suffix="__val")
"""

import os
import numpy as np
import pandas as pd


def plot_to_csv(plot_path: str, suffix: str = "") -> str:
    """
    Derive the CSV output path from a plot path.

    Maps  …/outputs/plots/…/name.png
    →     …/outputs/csv/…/name[suffix].csv

    Works with both forward- and back-slashes.
    """
    # Normalise separators
    p = plot_path.replace("\\", "/")

    # Replace the plots/ segment with csv/
    if "/plots/" in p:
        csv_path = p.replace("/plots/", "/csv/", 1)
    else:
        # Fallback: put csv/ next to plots/ at same level
        csv_path = p

    # Strip extension, add suffix + .csv
    csv_path = os.path.splitext(csv_path)[0] + suffix + ".csv"
    return csv_path


def csv_save(data, plot_path: str, suffix: str = "", index: bool = False) -> str:
    """
    Save *data* as a CSV file that mirrors *plot_path* in the csv/ tree.

    Parameters
    ----------
    data       : dict | list-of-dicts | pd.DataFrame | np.ndarray
                 The data to write.  Dicts are converted to DataFrames column-wise.
    plot_path  : str
                 Absolute or relative path of the corresponding .png file.
    suffix     : str
                 Optional panel label appended before ".csv", e.g. "__train".
    index      : bool
                 Whether to write the DataFrame row index (default False).

    Returns
    -------
    csv_path : str  — the path that was written
    """
    csv_path = plot_to_csv(plot_path, suffix)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            df = pd.DataFrame(data, columns=["value"])
        else:
            df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # Pad shorter arrays so all columns are the same length
        max_len = max((len(v) for v in data.values() if hasattr(v, "__len__")), default=1)
        padded = {}
        for k, v in data.items():
            arr = np.asarray(v).ravel()
            if len(arr) < max_len:
                arr = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
            padded[k] = arr
        df = pd.DataFrame(padded)
    else:
        df = pd.DataFrame(data)

    df.to_csv(csv_path, index=index)
    return csv_path
