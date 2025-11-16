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

# # 1) Run-Lookahead (replan every action)
# env = make_env()
# print("=== Run-Lookahead (every step) ===")
# logs_run = run_cage_controller(env, lazy_k=None, max_steps=10, verbose=False, use_coarse=False)
# # print("RUN", logs_run)
# summarize("RUN", logs_run)

# 2) Lazy-Lookahead (k=3)
env = make_env()
print("\n=== Lazy-Lookahead w/ Red Meander(k=3) ===")
logs_lazy = run_cage_controller(env, lazy_k=3, max_steps=50, verbose=True, use_coarse=False, red_policy=Meander_minimal())
# print("LAZY (k=3) w/ Red Meander", logs_lazy)
summarize("LAZY(k=3)", logs_lazy)

# # 3) Coarse + Lazy (k=3)
# env = make_env()
# print("\n=== Coarse + Lazy (k=3) ===")
# logs_coarse = run_cage_controller(env, lazy_k=3, max_steps=10, verbose=True, use_coarse=True)
# # print("COARSE+LAZY (k=3)", logs_coarse)
# summarize("COARSE+LAZY(k=3)", logs_coarse)

# 1) Run-Lookahead w/ Red Meander (replan every action)
env = make_env()
print("=== Run-Lookahead w/ Red Meander (every step) ===")
logs_run = run_cage_controller(env, lazy_k=None, max_steps=50, verbose=True, use_coarse=False, red_policy=Meander_minimal())
# print("RUN+RED_MEANDER", logs_run)
summarize("RUN+RED_MEANDER", logs_run)