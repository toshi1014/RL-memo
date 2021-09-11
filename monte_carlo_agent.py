from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
from environment import Action, Environment, show_reward_map

EPSILON = 0.1
MAX_EPISODE = 1000
GAMMA = 0.9


class MonteCarloAgent(Agent):
    def __init__(self, epsilon=EPSILON):
        super().__init__(epsilon)

    def learn(self, env, max_episode=MAX_EPISODE, gamma=GAMMA):
        self.Q = defaultdict(lambda: [0]*len(env.action_list))
        N = defaultdict(lambda: [0]*len(env.action_list))
        sum_reward_list = []

        for episode in range(max_episode):
            env.reset_state()
            log_list = []
            done = False
            sum_reward = 0

            while not done:
                state_now = env.state
                action_idx = self.policy(env.state, env.action_list)
                action = Action(env.action_list[action_idx])
                next_state, reward, done = env.step(action)

                log_list.append({"state": state_now, "action_idx": action_idx, "reward": reward})
                sum_reward += reward

            sum_reward_list.append(sum_reward)

            for idx, log in enumerate(log_list):
                state, action_idx = log["state"], log["action_idx"]

                G, t = 0, 0
                for j in range(idx, len(log_list)):
                    G += (gamma**t) * log_list[j]["reward"]
                    t += 1

                N[state][action_idx] += 1
                alpha = 1/N[state][action_idx]
                self.Q[state][action_idx] += alpha * (G - self.Q[state][action_idx])

            if (episode % 100) == 0:
                print("episode " + str(episode))

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.scatter(range(max_episode), sum_reward_list)
        plt.show()


def train():
    monte_carlo_agent = MonteCarloAgent()
    env = Environment()
    monte_carlo_agent.learn(env)
    monte_carlo_agent.test(env)
    show_reward_map(1, 1, env, MonteCarlo=monte_carlo_agent.Q)


if __name__ == '__main__':
    train()