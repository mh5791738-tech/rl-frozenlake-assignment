import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell('# Custom Reinforcement Learning Agent Implementation using Farama Gymnasium'),
    nbf.v4.new_markdown_cell('This notebook trains a Q-learning agent on the `FrozenLake-v1` environment and visualizes the learning curve and a short policy rollout.'),
    nbf.v4.new_code_cell(
        "import os\n"
        "import numpy as np\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "import gymnasium as gym\n"
        "from agent import QLearningAgent\n"
        "from utils import evaluate_policy, plot_training_curve, save_policy_gif\n\n"
        "np.random.seed(1)\n"
        "OUTPUT_DIR = os.getcwd()\n"
        "ENV_NAME = 'FrozenLake-v1'\n"
    ),
    nbf.v4.new_code_cell(
        "env = gym.make(ENV_NAME, is_slippery=False, render_mode='rgb_array')\n"
        "agent = QLearningAgent(\n"
        "    n_states=env.observation_space.n,\n"
        "    n_actions=env.action_space.n,\n"
        "    learning_rate=0.8,\n"
        "    discount_factor=0.99,\n"
        "    epsilon=1.0,\n"
        "    epsilon_min=0.05,\n"
        "    epsilon_decay=0.995,\n"
        ")\n"
        "rewards = agent.train(env, episodes=3000, max_steps=200)\n"
        "print('Episodes trained:', len(rewards))\n"
        "print('Average reward in last 100 episodes:', sum(rewards[-100:]) / 100)\n"
    ),
    nbf.v4.new_code_cell(
        "plot_training_curve(rewards, os.path.join(OUTPUT_DIR, 'training_progress.png'))\n"
        "plt.figure(figsize=(10, 5))\n"
        "plt.plot(rewards, alpha=0.3, label='Episode reward')\n"
        "plt.title('FrozenLake Training Curve')\n"
        "plt.xlabel('Episode')\n"
        "plt.ylabel('Reward')\n"
        "plt.grid(alpha=0.2)\n"
        "plt.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
    nbf.v4.new_code_cell(
        "avg_return, success_rate = evaluate_policy(env, agent.q_table, episodes=200)\n"
        "print(f'Average return over 200 evaluation episodes: {avg_return:.3f}')\n"
        "print(f'Success rate: {success_rate:.3f}')\n"
    ),
    nbf.v4.new_code_cell(
        "save_policy_gif(env, agent.q_table, os.path.join(OUTPUT_DIR, 'trained_agent.gif'), max_steps=200)\n"
        "print('Saved GIF to', os.path.join(OUTPUT_DIR, 'trained_agent.gif'))\n"
        "agent.save(os.path.join(OUTPUT_DIR, 'frozenlake_q_table.npy'))\n"
        "print('Saved model weights to', os.path.join(OUTPUT_DIR, 'frozenlake_q_table.npy'))\n"
    ),
    nbf.v4.new_code_cell("env.close()\nprint('Notebook completed successfully.')\n"),
]

with open('rl_assignment.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Notebook created: rl_assignment.ipynb')
