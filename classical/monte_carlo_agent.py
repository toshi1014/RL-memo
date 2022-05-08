import numpy as np
from collections import defaultdict
from environment import Experience
from agent import Agent


class MonteCarloAgent(Agent):
    def __init__(self, epsilon):
        super().__init__(epsilon)
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))
        self.N = defaultdict(lambda: defaultdict(lambda: 0))

    def learn(self, env, max_episodes, discount_rate):
        sum_reward_list = []

        for episode in range(max_episodes):
            done = False
            env.reset()
            sum_reward = 0
            experience_list = []

            while not done:
                state = env.state
                action = self.policy(state, env.action_list)
                next_state, reward, done = env.step(action)

                experience = Experience(
                    state, next_state, action, reward, done
                )
                experience_list.append(experience)
                sum_reward += reward

            sum_reward_list.append(sum_reward)

            for idx, experience in enumerate(experience_list):
                gain, t = 0, 0
                for j in range(idx, len(experience_list)):
                    gain += (discount_rate**t) * experience_list[j].reward
                    t += 1

                state = experience.state
                action = experience.action

                self.N[state][action] += 1
                learning_rate = 1 / self.N[state][action]
                self.Q[state][action] += learning_rate * \
                    (gain - self.Q[state][action])

            if episode % 10 == 0:
                print("Episode ", episode)

        return sum_reward_list