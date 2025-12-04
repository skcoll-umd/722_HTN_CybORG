# htn_agent.py

import time
from typing import Dict, Any
import numpy as np
from minicage_adapter import (
    translate_blue_obs_to_symbolic,
    encode_blue_action,
    make_blue_batch,
    get_red_action_from_state
)
from red_policy import NoOpRed

# ---- GTPyhop (global current_domain style) ----
from gtpyhop import State, Domain, declare_actions, find_plan, declare_task_methods

# Create a domain FIRST so current_domain is set
Domain("cage_htn")  # sets global current_domain internally

# ---------- constants ----------
# Use the same ordering as in minimal.py. These are the types of decoys that
# can be deployed on each host.
HOST_DECOYS = [
    [],  # def

    #ent
    [],
    ['Haraka', 'Tomcat', 'Vsftpd', 'Apache'],
    ['Femitter'],
    ['Femitter'],
    
    #ops
    [],  # ophost0
    [],  # ophost1
    [],  # ophost2
    ['Haraka', 'Apache', 'Tomcat', 'Vsftpd'],
    
    #user
    [], 
    ['Apache', 'Tomcat', 'SMSS', 'Svchost'],
    ['Femitter', 'Tomcat', 'Apache', 'SSHD'],
    ['Vsftpd', 'SSHD'],
    ['Vsftpd'],
]

HOST_ROLES = ['def', 'ent0', 'ent1', 'ent2',
              'ophost0', 'ophost1', 'ophost2', 'opserv',
              'user0', 'user1', 'user2', 'user3', 'user4']

CATEGORY_PRIORITY = {
    #Lower number is high priority; based on Cage Challenge 2 documentation
    'opserv': 0,   # opserv: -1 per turn + -10 impact → top priority
    'ent':      1,    # enterprise servers: -1 per turn
    'op':  2,    # operational hosts: -0.1 per turn
    'user':     3,    # user hosts: -0.1 per turn
    'def':      4,    # defender box last
    'other':    5,
}

FOOTHOLD_HOSTS = {"h12"}  # user4 / last host

# ---------- helpers ----------
def _copy(s: State) -> State:
    s2 = State('s')
    for k, v in s.__dict__.items():
        if isinstance(v, set): s2.__dict__[k] = set(v)
        elif isinstance(v, dict): s2.__dict__[k] = {kk:(set(vv) if isinstance(vv,set) else vv) for kk,vv in v.items()}
        else: s2.__dict__[k] = v
    return s2

def hostname(i: int) -> str:
    return f"h{i}"

def host_index(h: str) -> int:
    """Map 'h3' -> 3."""
    return int(h[1:]) if isinstance(h, str) and h.startswith('h') else int(h)

def host_role(h: str) -> str:
    """Map 'h3' -> 'ent2', etc."""
    return HOST_ROLES[host_index(h)]


def host_category(h: str) -> str:
    """High-level type: 'def', 'ent', 'op', 'opserv','user'."""
    r = host_role(h)
    if r.startswith('def'):
        return 'def'
    elif r.startswith('ent'):
        return 'ent'
    elif r.startswith('ophost'):
        return 'op'
    elif r.startswith('opserv'):
        return 'opserv'
    elif r.startswith('user'):
        return 'user'
    return 'other'

def host_priority_key(h: str):
    """
    Sort key for hosts:
      1) category importance (from rewards)
      2) host index (just for deterministic ordering).
    """
    cat = host_category(h)
    return (CATEGORY_PRIORITY.get(cat, 5), host_index(h))

def effective_compromised(s):
    # We only consider non-foothold compromised hosts as part of the goal
    return s.compromised - FOOTHOLD_HOSTS

def any_decoy_candidates(s: State) -> bool:
    """
    Return True iff there is at least one host where a decoy can still be placed.
    """
    if not hasattr(s, "decoys_remaining"):
        # If we don't track decoys yet in the planning state, be conservative:
        return True

    for h in s.hosts:
        idx = host_index(h)

        # This host type doesn't support decoys at all
        if not HOST_DECOYS[idx]:
            continue

        # Don't bother placing decoys on compromised hosts
        if h in s.compromised:
            continue

        # If there are any decoys left to place on this host
        if s.decoys_remaining.get(h, []):
            return True

    return False

# ---------- primitive actions ----------
def sleep(s: State):
    """
    Primitive 'sleep' action.
    Evironment: this is action 0 (no-op for Blue).
    Symbolically: treat it as doing nothing.
    """
    return s

def analyse_host(s: State, h: str):
    """
    Primitive 'analyse_host' on host h.
    Environment: gather info only (no direct state).
    Symbolically: we treat this as 'we have inspected h', so mark it scanned.
    """
    if h not in s.hosts:
        return False  # invalid action in this state

    # Keep memory that we've scanned this host
    s.scanned.add(h)
    return s

def place_decoy(s: State, h: str):
    """
    Primitive operator: place the highest-priority remaining decoy on host h.

    Assumptions:
      - s.decoys_remaining: dict[str, list[str]] mapping host -> remaining decoys
      - s.deployed_decoys:  dict[str, set[str]]  mapping host -> deployed decoys

    Effects (symbolic):
      - If host h has a remaining decoy, pop one from decoys_remaining[h]
        and add it to deployed_decoys[h].
      - If no decoys remain for h, this action fails (returns False) so the
        planner won’t rely on it.
    """

    # Host must be known in this state
    if not hasattr(s, "hosts") or h not in s.hosts:
        return False

    # We require decoy bookkeeping to exist (set up in the controller)
    if not hasattr(s, "decoys_remaining"):
        return False
    if not hasattr(s, "deployed_decoys"):
        return False

    remaining = s.decoys_remaining.get(h, [])
    if not remaining:
        # No decoys left for this host -> planning should treat this as invalid
        return False

    # Make a copy of the state (GTpyhop convention for operators)
    s2 = _copy(s)

    # Copy decoys_remaining dict and consume one decoy for h
    new_decoys = dict(s.decoys_remaining)
    host_list = list(remaining)
    next_decoy = host_list.pop(0)          # highest-priority decoy
    new_decoys[h] = host_list
    s2.decoys_remaining = new_decoys

    # Copy deployed_decoys and record the placed decoy
    new_deployed = {
        host: set(dec_set) for host, dec_set in s.deployed_decoys.items()
    }
    if h not in new_deployed:
        new_deployed[h] = set()
    new_deployed[h].add(next_decoy)
    s2.deployed_decoys = new_deployed

    return s2


def remove_processes(s: State, h: str):
    """
    Primitive 'remove_processes' on host h.
    Environment: tries to kill malicious processes (may fail if red has priv).
    Symbolically: we model this as 'attempt to clear compromise, but not as strong as restore'.
    A simple model: if h is compromised and not privileged, mark it not compromised.
    """
    if h not in s.hosts:
        return False

    # If we don't think it's compromised, there's nothing to remove.
    if h not in s.compromised:
        return s

    # Simple model: remove clears compromise but does not necessarily mark as 'patched'
    s.compromised.discard(h)

    return s


# ---- Restore host ----

def restore_host(s: State, h: str):
    """
    Primitive 'restore_host' on h.
    Environment: resets host to a known good state and penalizes Blue (-1 reward). We assume that
    any decoys placed on the host remain.
    Symbolically: we treat this as strong remediation — host is now patched & not compromised.
    """
    if h not in s.hosts:
        return False

    # After restore, we believe this host is clean.
    s.compromised.discard(h)
    s.patched.add(h)

    return s

# Register actions in current_domain
declare_actions(sleep, analyse_host, place_decoy, remove_processes, restore_host)

# ---------- HTN methods ----------
def m_secure_network_handle_compromised(s):
    """
    If there are any *non-foothold* compromised hosts, handle them first.
    This method should be tried BEFORE decoys/scans.
    """
    eff = effective_compromised(s)
    # print("[DEBUG M_SECURE_HANDLE] s.compromised =", s.compromised)
    # print("[DEBUG M_SECURE_HANDLE] effective_compromised =", eff)

    if not eff:
        return False  # no compromised to handle -> try next method

    # Handle compromised, then re-evaluate secure_network after that.
    return [
        ('handle_compromised',),
        ('secure_network',),
    ]


def m_secure_network_deploy_decoys(s: State):
    """
    Place decoys only if:
      - there are no non-foothold compromised hosts, AND
      - there is at least one host that can still take a decoy.
    """
    eff = effective_compromised(s)
    if eff:
        # If something important is compromised, don't waste time on decoys.
        return False

    if not any_decoy_candidates(s):
        # No decoy work left; move on to scans / done.
        return False

    return [
        ('deploy_decoys',),
        ('secure_network',),
    ]


def m_secure_network_scan(s):
    """
    If nothing to remediate or decoy, scan unscanned hosts (if any).
    """
    eff = effective_compromised(s)
    if eff:
        return False  # compromised still present; scanning is lower priority

    unscanned = s.hosts - s.scanned
    if not unscanned:
        # Nothing left to scan
        return False

    return [
        ('scan_hosts',),
        ('secure_network',),
    ]


def m_secure_network_done(s):
    """
    Base case: nothing left to do.

    Terminate when:
      - There are no non-foothold compromised hosts.
    (You can optionally add extra conditions like 'all_scanned' or 'no_decoys_left'.)
    """
    eff = effective_compromised(s)
    # print("[DEBUG M_SECURE_DONE] effective_compromised =", eff)

    if eff:
        # Still have non-foothold compromises -> not done
        return False

    # We accept that foothold hosts (e.g. h12) may remain in s.compromised.
    return []



def m_handle_compromised_high_value(s):
    """
    Handle high-value compromised hosts first:
      op_server (opserv) and ent* (enterprise servers).
    Strategy: remove + restore (we accept the -1 restore cost).
    """
    high_value = [h for h in effective_compromised(s)
                  if host_category(h) in ('opserv', 'ent')]

    if not high_value:
        return False

    # sort by category + index (reward-aware)
    high_value = sorted(high_value, key=host_priority_key)

    subtasks = []
    for h in high_value:
        subtasks.append(('remove_processes', h))
        subtasks.append(('restore_host', h))
        subtasks.append(('handle_compromised',))

    if not subtasks:
        return False

    return subtasks


def m_handle_compromised_low_value(s):
    """
    Handle remaining compromised hosts: op_host, user, def.
    Strategy: remove only (avoid paying restore cost unless necessary).
    """
    low_value = [h for h in effective_compromised(s)
                 if host_category(h) in ('op', 'user', 'def', 'other')]

    if not low_value:
        return False

    low_value = sorted(low_value, key=host_priority_key)

    subtasks = []
    for h in low_value:
        subtasks.append(('remove_processes', h))
        subtasks.append(('handle_compromised',))

    if not subtasks:
        return False

    return subtasks


def m_handle_compromised_none(s):
    """No compromised hosts → nothing to do."""
    if not effective_compromised(s):
        return []
    return False

def m_deploy_decoys_reward_aware(s):
    """
    Place a decoy on the highest-priority host that still supports decoys.
    Priority from reward structure: opserv > ent > op > user > def.
    """

    if not hasattr(s, "decoys_remaining"):
        s.decoys_remaining = {
            f"h{i}": list(HOST_DECOYS[i]) for i in range(len(HOST_DECOYS))
        }

    candidate_hosts = []
    for h in s.hosts:
        idx = host_index(h)

        if not HOST_DECOYS[idx]:
            continue

        if h in s.compromised:
            continue

        if s.decoys_remaining.get(h, []):
            candidate_hosts.append(h)

    if not candidate_hosts:
        return False

    # Highest priority host according to reward-aware ordering
    h_best = sorted(candidate_hosts, key=host_priority_key)[0]

    return [
        ('place_decoy', h_best),
    ]


def m_deploy_decoys_none(s):
    """No decoys left to place or no valid hosts."""
    return []

def m_scan_hosts_reward_aware(s):
    """
    Analyse one unscanned host, prioritised by reward:
      op_server > ent > op_host > user > def.
    """
    unscanned = s.hosts - s.scanned
    if not unscanned:
        return False

    h = sorted(unscanned, key=host_priority_key)[0]
    return [
        ('analyse_host', h),
    ]


def m_scan_hosts_none(s):
    """No unscanned hosts left."""
    if not (s.hosts - s.scanned):
        return []
    return False



#Register tasks with methods
declare_task_methods(
    'secure_network',
    m_secure_network_handle_compromised,
    m_secure_network_deploy_decoys,
    m_secure_network_scan,
    m_secure_network_done,
)

declare_task_methods(
    'handle_compromised',
    m_handle_compromised_high_value,
    m_handle_compromised_low_value,
    m_handle_compromised_none,
)

declare_task_methods(
    'deploy_decoys',
    m_deploy_decoys_reward_aware,
    m_deploy_decoys_none,
)

declare_task_methods(
    'scan_hosts',
    m_scan_hosts_reward_aware,
    m_scan_hosts_none,
)


# ---------- controller for SimplifiedCAGE ----------
def run_cage_controller(
    env,
    *,
    lazy_k=None,
    max_steps=20,
    verbose=True,
    use_coarse=False,
    red_policy=None
) -> Dict[str, Any]:
    """
    Run the Blue Agent HTN controller in the SimplifiedCAGE environment for a single episode.

    Planning modes (controlled by `lazy_k`):
      - Run-Lookahead (lazy_k is None):
          Replan after *every* primitive Blue action.
          This is the most reactive setting: each step calls the HTN planner
          on the latest symbolic belief state.

      - Run-Lazy-Lookahead (lazy_k is an int k >= 1):
          Replan only once every `k` primitive Blue actions.
          The controller:
            1) Calls the planner to get a full plan (sequence of primitive tasks).
            2) Executes up to `k` actions from that plan, *without* re-planning,
               unless a high-level task triggers an earlier replan.
            3) Re-enters the planner when k steps have been executed or the plan
               is exhausted.
          This reduces planning frequency at the cost of using a slightly stale plan.

    Episode horizon:
      - `max_steps` is a cap on the number of Blue actions in this episode.
        Once we have taken `max_steps` environment steps, the episode terminates,
        even if there is still a plan or compromised hosts remain.

    Environment contract (SimplifiedCAGE with num_envs=1):
      - state, info = env.reset()
      - next_state, reward, done, info = env.step(red_batch, blue_batch)
      - reward is a dict: {'Blue': (N, 1), 'Red': (N, 1)}

    Symbolic belief state:
      - We maintain a symbolic State `s` that is rebuilt from Blue's observation
        each step but also carries *memory* about what we've scanned, patched,
        and our decoy bookkeeping. This belief state is what the HTN planner
        reasons about.
    """
    # --- Reset env and build initial symbolic state ---
    state, info = env.reset()
    blue_obs = state['Blue'][0]               # (6 * num_nodes,)
    sym = translate_blue_obs_to_symbolic(blue_obs)

    s = State('s')
    s.__dict__.update(sym.__dict__)

    # Normalize to sets (defensive)
    s.scanned = set(getattr(s, "scanned", set()))
    s.patched = set(getattr(s, "patched", set()))
    s.compromised = set(getattr(s, "compromised", set()))
    s.hosts = set(getattr(s, "hosts", set()))

    # --- Initialize decoy bookkeeping ONCE for this episode ---
    # These are *planning-level* structures; the env doesn't expose decoys directly.
    if not hasattr(s, "decoys_remaining"):
        s.decoys_remaining = {
            f"h{i}": list(HOST_DECOYS[i]) for i in range(len(HOST_DECOYS))
        }
    if not hasattr(s, "deployed_decoys"):
        s.deployed_decoys = {
            f"h{i}": set() for i in range(len(HOST_DECOYS))
        }

    # --- Logs and goal ---
    logs = {
        'reward_total': 0.0,
        'steps': 0,
        'replans': 0,
        'plan_lengths': [],
        'plan_time_ms': [],
    }
    goal = ('secure_network',)

    num_envs, num_nodes = env.num_envs, env.num_nodes

    # Default red policy is a no-op red agent if none provided
    if red_policy is None:
        red_policy = NoOpRed()

    def plan_now():
        """
        Call the HTN planner on the current symbolic state `s`.
        """
        t0 = time.time()
        plan = find_plan(s, [goal]) or []   # uses global current_domain
        dt_ms = (time.time() - t0) * 1000.0
        logs['plan_time_ms'].append(dt_ms)
        logs['plan_lengths'].append(len(plan))

        if verbose:
            head = plan[:6]
            print(
                f"[PLAN] coarse={use_coarse} len={len(plan)} "
                f"{head}{'...' if len(plan) > 6 else ''}"
            )
        return plan

    # --- Main control loop over environment steps ---
    while logs['steps'] < max_steps:
        plan = plan_now()
        if not plan:
            # No plan found: either goal satisfied or we're stuck.
            if len(s.compromised) == 0 and verbose:
                print("[DONE] Goal satisfied (no compromised).")
            elif verbose:
                print("[STOP] No plan found; compromised remain:", s.compromised)
            break

        steps_since_check = 0

        # Execute the current plan (or its prefix), then replan.
        while plan and logs['steps'] < max_steps:
            task = plan.pop(0)

            # High-level task encountered -> trigger replan
            if task[0] in ('secure_network',):
                logs['replans'] += 1
                break

            # Primitive op and host
            op, h = task

            # Map primitive op to integer blue action for the env
            blue_int = encode_blue_action(op, h, num_nodes)
            blue_batch = make_blue_batch(blue_int, num_envs)

            # Get Red's action from its policy and current true state
            red_batch = get_red_action_from_state(
                state,
                red_policy=red_policy,
                num_envs=num_envs,
                verbose=verbose,
            )

            # --- Step the environment ---
            old_s = s
            next_state, reward, done, info = env.step(red_batch, blue_batch)

            # Build "obs-only" symbolic view from new Blue observation
            blue_obs = next_state['Blue'][0]
            sym_now = translate_blue_obs_to_symbolic(blue_obs)

            s_new = State('s')
            s_new.__dict__.update(sym_now.__dict__)

            # --- Merge beliefs (memory layer) ---

            # Ensure set types
            s_new.hosts = set(getattr(s_new, "hosts", set()))
            s_new.scanned = set(getattr(s_new, "scanned", set()))
            s_new.patched = set(getattr(s_new, "patched", set()))
            s_new.compromised = set(getattr(s_new, "compromised", set()))

            # Persistent facts: we remember what we've scanned and patched
            s_new.scanned |= getattr(old_s, "scanned", set())
            s_new.patched |= getattr(old_s, "patched", set())

            # Compromised is *not* persistent memory; it follows the current obs
            # (we already converted s_new.compromised from sym_now)

            # If we just analysed a host, mark it scanned in our belief
            if op == "analyse_host":
                s_new.scanned.add(h)

            # Carry decoy bookkeeping forward across time
            # (env doesn't expose this; it's purely a planning belief)
            if hasattr(old_s, "decoys_remaining"):
                s_new.decoys_remaining = old_s.decoys_remaining
            if hasattr(old_s, "deployed_decoys"):
                s_new.deployed_decoys = old_s.deployed_decoys

            # Commit new symbolic + true env state
            s = s_new
            state = next_state

            # --- Logging rewards and steps ---
            rew_blue = float(np.array(reward['Blue']).reshape(-1)[0])
            logs['reward_total'] += rew_blue
            logs['steps'] += 1

            if verbose:
                print(
                    f"[ACT] {task} -> r_blue={rew_blue:.3f} | "
                    f"compromised={len(s.compromised)}"
                )

            # Check if it's time to replan (Run-Lookahead vs Run-Lazy-Lookahead)
            steps_since_check += 1
            need_replan = (lazy_k is None) or (steps_since_check >= lazy_k)
            if need_replan:
                logs['replans'] += 1
                break

    return logs
