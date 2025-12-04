# run_simplified_cage.py
from htn_agent import run_cage_controller
from mini_CAGE.minimal import SimplifiedCAGE
from mini_CAGE.test_agent import Meander_minimal, B_line_minimal

def make_env():
    return SimplifiedCAGE(num_envs=1, num_nodes=13, remove_bugs=False)

def summarize(label: str, logs: dict):
    plan_lengths = logs.get("plan_lengths", []) or []
    plan_times   = logs.get("plan_time_ms", []) or []
    avg_len  = (sum(plan_lengths) / len(plan_lengths)) if plan_lengths else 0.0
    avg_time = (sum(plan_times)   / len(plan_times))   if plan_times   else 0.0
    print(
        f"{label} → steps={logs.get('steps',0)}, replans={logs.get('replans',0)}, "
        f"reward_total={logs.get('reward_total',0.0):.3f}, "
        f"avg_plan_len={avg_len:.2f}, avg_plan_time_ms={avg_time:.2f}"
    )

# 1) Run-Lookahead w/ Red Meander (replan every action)
env = make_env()
print("=== Run-Lookahead (every step) w/ Red Meander ===")
logs_run = run_cage_controller(env, lazy_k=None, max_steps=50, verbose=False, use_coarse=False, red_policy=Meander_minimal())
# print("RUN", logs_run)
summarize("RUN", logs_run)

# 2) Lazy-Lookahead (k=2) w/ Red Meander
env = make_env()
print("\n=== Lazy-Lookahead w/ Red Meander(k=2) ===")
logs_lazy = run_cage_controller(env, lazy_k=2, max_steps=50, verbose=False, use_coarse=False, red_policy=Meander_minimal())
# print("LAZY (k=2) w/ Red Meander", logs_lazy)
summarize("LAZY(k=3)", logs_lazy)

# 3) Lazy-Lookahead (k=4) w/ Red Meander
env = make_env()
print("\n=== Lazy-Lookahead w/ Red Meander(k=4) ===")
logs_lazy = run_cage_controller(env, lazy_k=4, max_steps=50, verbose=False, use_coarse=False, red_policy=Meander_minimal())
# print("LAZY (k=4) w/ Red Meander", logs_lazy)
summarize("LAZY(k=3)", logs_lazy)

# 3) Lazy-Lookahead (k=8) w/ Red Meander
env = make_env()
print("\n=== Lazy-Lookahead w/ Red Meander(k=8) ===")
logs_lazy = run_cage_controller(env, lazy_k=8, max_steps=50, verbose=False, use_coarse=False, red_policy=Meander_minimal())
# print("LAZY (k=8) w/ Red Meander", logs_lazy)
summarize("LAZY(k=8)", logs_lazy)

# 3) Lazy-Lookahead (k=12) w/ Red Meander
env = make_env()
print("\n=== Lazy-Lookahead w/ Red Meander(k=12) ===")
logs_lazy = run_cage_controller(env, lazy_k=12, max_steps=50, verbose=False, use_coarse=False, red_policy=Meander_minimal())
# print("LAZY (k=8) w/ Red Meander", logs_lazy)
summarize("LAZY(k=12)", logs_lazy)
#==============================================================

# 1) Run-Lookahead w/ BLine (replan every action)
env = make_env()
print("=== Run-Lookahead (every step) w/ BLine ===")
logs_run = run_cage_controller(env, lazy_k=None, max_steps=50, verbose=False, use_coarse=False, red_policy=B_line_minimal())
# print("RUN", logs_run)
summarize("RUN", logs_run)

# 2) Lazy-Lookahead (k=2) w/ Bline
env = make_env()
print("\n=== Lazy-Lookahead w/ Bline(k=2) ===")
logs_lazy = run_cage_controller(env, lazy_k=2, max_steps=50, verbose=False, use_coarse=False, red_policy=B_line_minimal())
# print("LAZY (k=2) w/ Bline", logs_lazy)
summarize("LAZY(k=3)", logs_lazy)

# 3) Lazy-Lookahead (k=4) w/ Bline
env = make_env()
print("\n=== Lazy-Lookahead w/ Bline(k=4) ===")
logs_lazy = run_cage_controller(env, lazy_k=4, max_steps=50, verbose=False, use_coarse=False, red_policy=B_line_minimal())
# print("LAZY (k=4) w/ Bline", logs_lazy)
summarize("LAZY(k=3)", logs_lazy)

# 3) Lazy-Lookahead (k=8) w/ Bline
env = make_env()
print("\n=== Lazy-Lookahead w/ Bline(k=8) ===")
logs_lazy = run_cage_controller(env, lazy_k=8, max_steps=50, verbose=False, use_coarse=False, red_policy=B_line_minimal())
# print("LAZY (k=8) w/ Bline", logs_lazy)
summarize("LAZY(k=8)", logs_lazy)

# 4) Lazy-Lookahead (k=12) w/ Bline
env = make_env()
print("\n=== Lazy-Lookahead w/ Bline(k=12) ===")
logs_lazy = run_cage_controller(env, lazy_k=12, max_steps=50, verbose=False, use_coarse=False, red_policy=B_line_minimal())
# print("LAZY (k=12) w/ Bline", logs_lazy)
summarize("LAZY(k=12)", logs_lazy)