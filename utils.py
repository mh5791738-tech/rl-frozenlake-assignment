import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def evaluate_policy(env, q_table, episodes=200):
    wins = 0
    total_return = 0.0

    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0

        while True:
            action = int(np.argmax(q_table[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break

        total_return += total_reward
        if total_reward > 0:
            wins += 1

    avg_return = total_return / episodes
    success_rate = wins / episodes
    return avg_return, success_rate


def plot_training_curve(rewards, save_path):
    if not rewards:
        raise ValueError("No rewards available to plot")

    window = max(1, len(rewards) // 20)
    rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    x_axis = np.arange(len(rolling_avg))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.35, label="Episode Reward")
    ax.plot(x_axis, rolling_avg, color="tab:orange", linewidth=2, label=f"{window}-Episode Moving Average")
    ax.set_title("FrozenLake Q-Learning Training Progress")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _render_frozenlake_frame(env, state, cell_size=80):
    desc = env.unwrapped.desc.copy()
    rows, cols = desc.shape
    width = cols * cell_size
    height = rows * cell_size
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)

    for row in range(rows):
        for col in range(cols):
            tile = desc[row, col]
            if tile == b"S":
                color = (175, 255, 175)
            elif tile == b"F":
                color = (225, 225, 225)
            elif tile == b"H":
                color = (55, 55, 55)
            elif tile == b"G":
                color = (255, 220, 90)
            else:
                color = (255, 255, 255)

            x0, y0 = col * cell_size, row * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color)
            draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)

    agent_row, agent_col = divmod(state, cols)
    cx = agent_col * cell_size + cell_size / 2
    cy = agent_row * cell_size + cell_size / 2
    radius = int(cell_size * 0.22)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 0, 0))
    return np.array(image)


def save_policy_gif(env, q_table, save_path, max_steps=200):
    frames = []
    state, _ = env.reset()

    for _ in range(max_steps):
        frames.append(_render_frozenlake_frame(env, state))
        action = int(np.argmax(q_table[state]))
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state
        if terminated or truncated:
            frames.append(_render_frozenlake_frame(env, state))
            break

    if not frames:
        raise ValueError("No frames were captured for GIF generation")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, fps=5)
