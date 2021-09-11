import numpy as np

class Agent():
    def __init__(self, epsilon_):
        self.Q = {}
        self.epsilon = epsilon_

    def policy(self, state, action_list):
        action_idx = np.argmax(self.Q[state])

        if (np.random.random() < self.epsilon) | (sum(self.Q[state]) == 0):
            action_idx = np.random.randint(len(action_list))

        return action_idx

    def test(self, env):
        env.reset_state()
        done = False
        sum_reward = 0
        
        while not done:
            action_idx = np.argmax(self.Q[env.state])
            next_state, reward, done = env.step(env.action_list[action_idx])
            sum_reward += reward
            print(next_state)

        print("reward:", sum_reward)
