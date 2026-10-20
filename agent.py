import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        learning_rate=0.8,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)

    def act(self, state):
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, done):
        best_next_value = 0.0 if done else np.max(self.q_table[next_state])
        target = reward + self.discount_factor * best_next_value
        self.q_table[state, action] += self.learning_rate * (
            target - self.q_table[state, action]
        )

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes=3000, max_steps=200):
        rewards = []

        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            episode_reward = 0.0

            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                self.learn(state, action, reward, next_state, terminated or truncated)
                state = next_state

                if terminated or truncated:
                    break

            rewards.append(episode_reward)
            self.update_epsilon()

        return rewards

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)
        self.n_states, self.n_actions = self.q_table.shape
