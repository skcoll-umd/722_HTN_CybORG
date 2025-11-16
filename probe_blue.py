from mini_CAGE.minimal import SimplifiedCAGE  # wherever your env is
from minicage_adapter import make_blue_batch, make_red_noop
import numpy as np

# paste the probe function here
def probe_blue_actions(env, num_nodes):
    state, info = env.reset()
    num_envs = env.num_envs

    for action_int in range(0, 52):
        blue_batch = make_blue_batch(action_int, num_envs)
        red_batch = make_red_noop(num_envs)

        prev_row = state['Blue'][0].copy()
        next_state, reward, done, info = env.step(red_batch, blue_batch)
        next_row = next_state['Blue'][0]

        mat_prev = prev_row.reshape(num_nodes, 6)
        mat_next = next_row.reshape(num_nodes, 6)

        changed = np.where((mat_prev != mat_next).any(axis=1))[0]
        if len(changed) > 0:
            print(f"ACTION {action_int} changes hosts {changed}:")
            for i in changed:
                print(f"  h{i}: {mat_prev[i]} -> {mat_next[i]}")

        state = next_state
        if done:
            break


if __name__ == "__main__":
    env = SimplifiedCAGE(num_envs=1)
    num_nodes = env.num_nodes
    probe_blue_actions(env, num_nodes)
