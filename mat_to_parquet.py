#!/usr/bin/env python3
"""
Convert MATLAB .mat files to Apache Parquet format.
Handles struct arrays, cell arrays, and nested structures.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import io
import warnings

warnings.filterwarnings('ignore')

try:
    import mat73
    HAS_MAT73 = True
except ImportError:
    HAS_MAT73 = False


def _extract_value(item):
    """Recursively convert MATLAB types to standard Python types."""
    # Handle mat_struct (scipy.io.loadmat with struct_as_record=False)
    if type(item).__name__ == 'mat_struct':
        # Extract all attributes that don't start with '_'
        return {k: _extract_value(getattr(item, k)) for k in dir(item) if not k.startswith('_')}
    
    # Handle numpy arrays
    if isinstance(item, np.ndarray):
        if item.size == 0:
            return None
        if item.size == 1:
            return _extract_value(item.item())
        # If it's a record array (struct array)
        if item.dtype.names:
            return [_extract_value(row) for row in item]
        # Standard array -> list
        return item.tolist()
    
    # Handle lists/tuples (cell arrays)
    if isinstance(item, (list, tuple)):
        return [_extract_value(x) for x in item]
    
    # Handle numpy scalars
    if isinstance(item, (np.integer, np.floating, np.bool_)):
        return item.item()
    
    return item


def flatten_dict(d, prefix=''):
    """Flatten a nested dictionary into a single level."""
    out = {}
    for a, b in d.items():
        key = f"{prefix}{a}" if prefix else a
        if isinstance(b, dict):
            out.update(flatten_dict(b, prefix=key + '_'))
        else:
            out[key] = b
    return out


def process_variable(name, data):
    """Convert a single MAT variable into a tabular format."""
    # 1. Normalize data to a Python list of dictionaries or a simple list
    val = _extract_value(data)
    
    # Case A: It's a list of dictionaries (struct array)
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
        # Flatten each dictionary in the list
        flattened_list = [flatten_dict(item) for item in val]
        return pd.DataFrame(flattened_list)
    
    # Case B: It's a single dictionary (single struct)
    if isinstance(val, dict):
        flat = flatten_dict(val)
        # Return as a single-row DataFrame
        return pd.DataFrame([flat])
    
    # Case C: It's a simple list/array
    if isinstance(val, list):
        return pd.DataFrame({name: val})
    
    # Case D: It's a scalar
    return pd.DataFrame({name: [val]})


def load_mat_file(filepath):
    filepath = Path(filepath)
    try:
        # We use struct_as_record=False to get mat_struct objects, 
        # which we then handle in _extract_value
        data = io.loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in data.items() if not k.startswith('__')}
    except Exception as e:
        if not HAS_MAT73:
            print(f"Error: Could not load with scipy and mat73 not installed.")
            sys.exit(1)
        try:
            return mat73.loadmat(str(filepath))
        except Exception as e2:
            print(f"Error loading {filepath}: {e} / {e2}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Convert MATLAB .mat files to Parquet')
    parser.add_argument('matfile', help='Input .mat file')
    parser.add_argument('-o', '--output', help='Output parquet file')
    parser.add_argument('-c', '--compression', default='snappy', choices=['snappy', 'gzip', 'brotli', 'zstd'])
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Loading {args.matfile}...")
    data = load_mat_file(args.matfile)
    
    all_dfs = []
    for var_name, var_data in data.items():
        if args.verbose:
            print(f"Processing variable: {var_name}")
        try:
            df = process_variable(var_name, var_data)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {var_name}: {e}")

    if not all_dfs:
        print("No valid data found.")
        sys.exit(1)
    
    # Try to merge if they all have the same length
    if len(all_dfs) > 1:
        lengths = [len(df) for df in all_dfs]
        if len(set(lengths)) == 1:
            result = pd.concat(all_dfs, axis=1)
        else:
            # Fallback: just take the first one or save separately
            # For simplicity, we merge what we can and warn
            print("Warning: Variables have different lengths. Only merging first variable.")
            result = all_dfs[0]
    else:
        result = all_dfs[0]
    
    out_path = args.output or f"{Path(args.matfile).stem}.parquet"
    if args.verbose:
        print(f"Writing to {out_path} ({result.shape[0]} rows, {result.shape[1]} cols)...")
    
    result.to_parquet(out_path, compression=args.compression, index=False)
    print(f"✓ Successfully converted to {out_path}")


if __name__ == '__main__':
    sys.exit(main())
