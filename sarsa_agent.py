from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from environment import Action, Environment, show_reward_map


EPSILON = 0.1
MAX_EPISODE = 1000
GAMMA = 0.9
LEARNING_RATE = 0.1

class SARSAAgent(Agent):
    def __init__(self, epsilon=EPSILON):
        super().__init__(epsilon)

    def learn(self, env, max_episode=MAX_EPISODE, gamma=GAMMA, learning_rate=LEARNING_RATE):
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))
        reward_log = []

        for episode in range(max_episode):
            env.reset_state()
            done = False
            sum_reward = 0

            while not done:
                state_now = env.state
                action = self.policy(env.state, env.action_list)
                next_state, reward, done = env.step(action)

                next_action = self.policy(next_state, env.action_list)
                gain = reward + gamma*self.Q[next_state][next_action]

                self.Q[state_now][action] += learning_rate * (gain - self.Q[state_now][action])

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
    show_reward_map(1, 1, env, SARSA=sarsa_agent.Q)


if __name__ == '__main__':
    train()