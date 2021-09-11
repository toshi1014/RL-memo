import numpy as np
from environment import Environment

class Agent():
    def __init__(self, epsilon_):
        self.Q = {}
        self.epsilon = epsilon_

    def q_state_init(self, state, arg_action_list):
        if not self.Q[state]:
            for action in arg_action_list:
                self.Q[state][action]

    def policy(self, state, arg_action_list):
        self.q_state_init(state, arg_action_list)

        action_list = []
        value_list = []
        for action in self.Q[state]:
            action_list.append(action)
            value_list.append(self.Q[state][action])

        action = action_list[np.argmax(value_list)]

        if (np.random.random() < self.epsilon) | (sum(self.Q[state].values()) == 0):
            action = np.random.choice(action_list)

        return action

    def test(self, env):
        env.reset_state()
        done = False
        sum_reward = 0

        while not done:
            action = self.policy(env.state, env.action_list)
            next_state, reward, done = env.step(action)
            sum_reward += reward
            print(next_state)

        print("reward:", sum_reward)