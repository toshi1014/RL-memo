import numpy as np


class Agent:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def q_state_init(self, state, arg_action_list):
        if not self.Q[state]:
            for action in arg_action_list:
                self.Q[state][action]

    def policy(self, state, action_list, greedy=False):
        self.q_state_init(state, action_list)
        bool_first_state = sum(self.Q[state].values()) == 0

        if ((np.random.random() < self.epsilon) | bool_first_state) & (not greedy):
            return np.random.choice(action_list)
        else:
            action_idx = np.argmax(list(self.Q[state].values()))
            return action_list[action_idx]

    def test(self, env):
        done = False
        sum_reward = 0
        env.reset()

        while not done:
            state = env.state
            action = self.policy(state, env.action_list, greedy=True)
            next_state, reward, done = env.step(action)
            print(action, next_state)
            sum_reward += reward

        return sum_reward