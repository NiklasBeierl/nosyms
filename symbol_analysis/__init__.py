import json
from collections import defaultdict
from typing import List, Iterable, Dict, Tuple
import pandas as pd
import numpy as np


def json_load(p):
    with open(p) as f:
        return json.load(f)


def get_user_type_sizes(syms: Dict) -> Dict[str, int]:
    return {name: ut["size"] for name, ut in syms["user_types"].items()}


_SYMBOLS_BATCH_STATS_COLS = {
    "num_user_types": lambda syms: len(syms["user_types"]),
    "max_user_type_size": lambda syms: max(get_user_type_sizes(syms).values()),
    "median_user_type_size": lambda syms: np.median(list(get_user_type_sizes(syms).values())),
    "mean_user_type_size": lambda syms: np.mean(list(get_user_type_sizes(syms).values())),
}


def get_symbols_batch_stats(paths: Iterable[str]) -> pd.DataFrame:
    rows: List[Tuple] = []
    for path in paths:
        syms = json_load(path)
        stats = tuple(f(syms) for f in _SYMBOLS_BATCH_STATS_COLS.values())
        rows.append((path,) + stats)
    return pd.DataFrame(data=rows, columns=("path",) + tuple(_SYMBOLS_BATCH_STATS_COLS.keys()))


_SYMBOLS_BATCH_USER_TYPE_STATS_COLS = {
    "max_len": lambda sizes: max(sizes),
    "min_len": lambda sizes: min(sizes),
    "mean_len": lambda sizes: np.mean(sizes),
    "occurrences": lambda sizes: len(sizes),
}


def get_symbols_batch_user_type_stats(paths: Iterable[str]) -> pd.DataFrame:
    user_type_sizes = defaultdict(list)
    for path in paths:
        syms = json_load(path)
        ut_sizes = get_user_type_sizes(syms)
        for k, v in ut_sizes.items():
            user_type_sizes[k].append(v)
    rows: List[Tuple] = []
    for name, sizes in user_type_sizes.items():
        rows.append((name,) + tuple(f(sizes) for f in _SYMBOLS_BATCH_USER_TYPE_STATS_COLS.values()))
    df = pd.DataFrame(data=rows, columns=("name",) + tuple(_SYMBOLS_BATCH_USER_TYPE_STATS_COLS.keys()))
    df.sort_values("mean_len", inplace=True)
    return df
