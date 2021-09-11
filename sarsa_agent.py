from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment, show_reward_map


EPSILON = 0.1
MAX_EPISODE = 1000
GAMMA = 0.9
LEARNING_RATE = 0.1

class SARSAAgent(Agent):
    def __init__(self, epsilon=EPSILON):
        super().__init__(epsilon)

    def learn(self, env, max_episode=MAX_EPISODE, gamma=GAMMA, learning_rate=LEARNING_RATE):
        self.Q = defaultdict(lambda: [0]*len(env.action_list))
        reward_log = []

        for episode in range(max_episode):
            env.reset_state()
            done = False
            sum_reward = 0

            while not done:
                state_now = env.state
                action_idx = self.policy(env.state, env.action_list)
                next_state, reward, done = env.step(env.action_list[action_idx])

                next_action_idx = self.policy(next_state, env.action_list)
                gain = reward + gamma*self.Q[next_state][next_action_idx]

                self.Q[state_now][action_idx] += learning_rate * (gain - self.Q[state_now][action_idx])

                sum_reward += reward

            reward_log.append(sum_reward)

            if episode % 100 == 0:
                print("episode " + str(episode))

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.scatter(range(max_episode), reward_log)
        plt.show()


def train():
    sarsa_agent = SARSAAgent()
    env = Environment()
    sarsa_agent.learn(env)
    sarsa_agent.test(env)
    show_reward_map(env, SARSA=sarsa_agent.Q)


if __name__ == '__main__':
    train()