import os

import gymnasium as gym
import numpy as np

from agent import QLearningAgent
from utils import evaluate_policy, plot_training_curve, save_policy_gif


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(OUTPUT_DIR, "frozenlake_q_table.npy")
GIF_PATH = os.path.join(OUTPUT_DIR, "trained_agent.gif")
PLOT_PATH = os.path.join(OUTPUT_DIR, "training_progress.png")


def main():
    np.random.seed(1)
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.8,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )

    rewards = agent.train(env, episodes=3000, max_steps=200)
    agent.save(MODEL_PATH)
    plot_training_curve(rewards, PLOT_PATH)

    avg_return, success_rate = evaluate_policy(env, agent.q_table, episodes=200)
    save_policy_gif(env, agent.q_table, GIF_PATH, max_steps=200)

    print(f"Training episodes: 3000")
    print(f"Average return over 200 evaluation episodes: {avg_return:.3f}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Q-table saved to: {MODEL_PATH}")
    print(f"GIF saved to: {GIF_PATH}")
    print(f"Training plot saved to: {PLOT_PATH}")

    env.close()


if __name__ == "__main__":
    main()
