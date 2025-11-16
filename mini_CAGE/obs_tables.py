# mini_CAGE/obs_tables.py
# Utilities to decode & pretty-print MiniCAGE observations for Blue.

from typing import List, Optional, Tuple
import numpy as np

# Pull canonical host order from the env implementation
from mini_CAGE.minimal import HOSTS  # ['def','ent0','ent1',...,'user4']

# Per-host column meanings for the Blue observation vector
BLUE_COLS: List[str] = [
    "activity_scan",              # 1 if scan on this host THIS step
    "activity_exploit_detected",  # 1 if exploit detected on this host THIS step
    "safe_flag",                  # 1 if analyse(priv.) or remove indicates safety
    "compromised_flag",           # 1 if host believed compromised
    "scan_recent_0_2",            # 0/1/2 (2 = scanned this step, decays to 1)
    "decoys_count",               # number of decoys on this host
]

def _as_row(obs: np.ndarray) -> np.ndarray:
    """Accept (1, N) or (N,) and return (N,) row."""
    a = np.asarray(obs)
    if a.ndim == 2 and a.shape[0] == 1:
        return a[0]
    if a.ndim == 1:
        return a
    raise ValueError(f"Expected (N,) or (1,N) obs, got shape {a.shape}")

def blue_obs_grid(blue_obs_row: np.ndarray) -> np.ndarray:
    """
    Reshape the Blue observation flat vector into (num_hosts, 6) grid.
    Returns an array of shape (len(HOSTS), len(BLUE_COLS)).
    """
    row = _as_row(blue_obs_row).ravel()
    H, C = len(HOSTS), len(BLUE_COLS)
    expected = H * C
    if row.size != expected:
        raise ValueError(f"Blue obs length mismatch: expected {expected} (= {H}*{C}), got {row.size}")
    return row.reshape(H, C)

def format_blue_table(blue_obs_row: np.ndarray, max_rows: Optional[int] = None) -> str:
    """
    Return a pretty string table for Blue obs with host and BLUE_COLS headers.
    """
    grid = blue_obs_grid(blue_obs_row)
    rows = len(HOSTS) if max_rows is None else min(len(HOSTS), max_rows)

    header = ["host"] + BLUE_COLS
    # choose simple column widths
    widths = [max(5, len(h)) for h in header]

    # build lines
    lines = []
    lines.append("[Blue] observation (labeled)")
    lines.append(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    lines.append("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for i in range(rows):
        row_vals = []
        for j, x in enumerate(grid[i]):
            # all these features are effectively small integers
            try:
                row_vals.append(str(int(x)))
            except Exception:
                row_vals.append(str(x))
        row = [HOSTS[i]] + row_vals
        lines.append(" | ".join(s.ljust(w) for s, w in zip(row, widths)))

    if rows < len(HOSTS):
        lines.append(f"... ({len(HOSTS) - rows} more hosts)")
    return "\n".join(lines)

def print_blue_table(blue_obs_row: np.ndarray, max_rows: Optional[int] = None) -> None:
    """Print the same table returned by format_blue_table()."""
    print(format_blue_table(blue_obs_row, max_rows=max_rows))
