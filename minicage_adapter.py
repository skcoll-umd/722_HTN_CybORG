# minicage_adapter.py
from dataclasses import dataclass
from typing import Any, Dict, Set, Optional, Union, Callable
import numpy as np

# Blue_obs is (num_envs, 6*num_nodes) with per-host 6 cols:
# [activity_scan, activity_exploit_detected, safe_flag, compromised_flag, scan_recent_0_2, decoys_count]
BLUE_COLS = [
    "activity_scan",
    "activity_exploit_detected",
    "safe_flag",
    "compromised_flag",
    "scan_recent_0_2",
    "decoys_count",
]

@dataclass
class Symbolic:
    hosts: Set[str]
    compromised: Set[str]
    isolated: Set[str]   # not explicit in env; we leave empty and let HTN ignore it
    patched: Set[str]    # we treat safe_flag=1 as "patched/clean"
    scanned: Set[str]    # recent scan > 0
    services: Dict[str, Set[str]]  # empty for now
    subnets: Dict[str, Set[str]]

def _hostnames(n: int):
    return [f"h{i}" for i in range(n)]

def translate_blue_obs_to_symbolic(blue_obs_row: np.ndarray) -> Symbolic:
    """
    blue_obs_row: shape (6*num_nodes,) -- i.e., for a single env row.
    """
    # infer num_nodes
    if blue_obs_row.ndim != 1:
        blue_obs_row = blue_obs_row.reshape(-1)
    total = blue_obs_row.shape[0]
    assert total % 6 == 0, f"Expected 6*num_nodes columns; got {total}"
    num_nodes = total // 6
    H = _hostnames(num_nodes)

    # reshape to [num_nodes, 6]
    mat = blue_obs_row.reshape(num_nodes, 6)

    idx = {name: i for i, name in enumerate(BLUE_COLS)}
    compromised_now = {
        H[i] for i in range(num_nodes) if mat[i, idx["compromised_flag"]] > 0
    }
    scanned_now = {
        H[i] for i in range(num_nodes) if mat[i, idx["scan_recent_0_2"]] > 0
        or mat[i, idx["activity_scan"]] > 0
    }
    patched_now = {
        H[i] for i in range(num_nodes) if mat[i, idx["safe_flag"]] > 0
    }

    hosts = set(H)
    subnets = {"default": set(H)}
    return Symbolic(
        hosts=hosts,
        compromised=compromised_now,
        isolated=set(),      # env doesn't track this; HTN can ignore/handle gracefully
        patched=patched_now,
        scanned=scanned_now,
        services={},
        subnets=subnets,
    )

def host_to_index(h: str) -> int:
    # expects names like "h0", "h1", ...
    if isinstance(h, str) and h.startswith("h"):
        return int(h[1:])
    return int(h)

def encode_blue_action(op: str, h: str, num_nodes: int) -> int:
    """
    Map primitive op on host h -> integer blue action as required by SimplifiedCAGE.

    Encoding from minimal.py (update_blue):
      action 0          : sleep / NOOP
      action 1..N       : analyse(host 0..N-1)
      action N+1..2N    : decoy(host)
      action 2N+1..3N   : remove(host)
      action 3N+1..4N   : restore(host)
    """
    i = host_to_index(h)  # h like "h0","h1",...

    if op == 'sleep':
        return 0

    if op == 'analyse_host':
        # action_alloc = 0  -> actions 1..N
        return 1 + i

    elif op == 'place_decoy':  # if/when you add this to HTN
        # action_alloc = 1  -> actions N+1..2N
        return 1 + num_nodes + i

    elif op == 'remove_processes':
        # action_alloc = 2  -> actions 2N+1..3N
        return 1 + 2 * num_nodes + i

    elif op == 'restore_host':
        # action_alloc = 3  -> actions 3N+1..4N
        return 1 + 3 * num_nodes + i

    else:
        raise ValueError(f"Unknown primitive op for blue: {op}")


def make_red_noop(num_envs: int) -> np.ndarray:
    """
    Red action array of zeros (no-op w.r.t. the step branching).
    Shape must be (num_envs, 1) to satisfy asserts.
    """
    return np.zeros((num_envs, 1), dtype=int)

def make_blue_batch(action_int: int, num_envs: int) -> np.ndarray:
    """
    Blue action needs shape (num_envs, 1).
    """
    return np.full((num_envs, 1), action_int, dtype=int)

# --- Red policy integration (MiniCAGE) ---------------------------------------

def normalize_red_batch(action, num_envs: int) -> np.ndarray:
    """
    Ensure Red action has shape (num_envs, 1) of dtype int.
    Accepts int, 1D, or 2D arrays (or lists). Broadcast if needed.
    """
    a = np.asarray(action)
    if a.ndim == 0:                       # scalar -> broadcast
        a = np.full((num_envs, 1), int(a), dtype=int)
    elif a.ndim == 1:                     # (num_envs,) -> (num_envs,1)
        if a.size == 1 and num_envs > 1:  # single value, broadcast
            a = np.full((num_envs, 1), int(a.item()), dtype=int)
        else:
            a = a.reshape(-1, 1).astype(int, copy=False)
    elif a.ndim == 2:
        # keep (num_envs, 1) if already correct; otherwise coerce 2D to that
        if a.shape[1] != 1:
            a = a[:, :1]
        a = a.astype(int, copy=False)
    else:
        raise ValueError(f"Unexpected Red action shape: {a.shape}")
    if a.shape[0] != num_envs:
        # final safety broadcast
        if a.shape == (1, 1):
            a = np.full((num_envs, 1), int(a.item()), dtype=int)
        else:
            raise ValueError(f"Red action batch size {a.shape[0]} != num_envs {num_envs}")
    return a

def get_red_action_from_state(
    state: Dict[str, Any],
    *,
    red_policy: Optional[Union[Callable, object]],
    num_envs: int,
    verbose: bool = False,
) -> np.ndarray:
    """
    Adapter entry point: build a (num_envs,1) Red action from the env `state`
    using the provided `red_policy` (function or object with .get_action).
    Falls back to NOOP on error or if None.
    """
    if red_policy is None:
        return make_red_noop(num_envs)

    try:
        red_obs = state["Red"]  # expected MiniCAGE Red observation
        # Call policy with flexible signatures
        if hasattr(red_policy, "get_action"):
            try:
                action = red_policy.get_action(red_obs, state)
            except TypeError:
                action = red_policy.get_action(red_obs)
        else:
            try:
                action = red_policy(red_obs, state)
            except TypeError:
                action = red_policy(red_obs)

        return normalize_red_batch(action, num_envs)
    except Exception as e:
        if verbose:
            print(f"[WARN][adapter] Red policy error: {e}. Using NOOP.")
        return make_red_noop(num_envs)



