import numpy as np
from collections import defaultdict, deque
from environment import Experience
from agent import Agent


class QLearningAgent(Agent):
    def __init__(self, epsilon, multi_step_window):
        super().__init__(epsilon)
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))
        self.multi_step_experience = deque(maxlen=multi_step_window)
        self.multi_step_window = multi_step_window

    def learn(self, env, max_episodes, discount_rate, learning_rate):
        sum_reward_list = []

        for episode in range(max_episodes):
            done = False
            env.reset()
            sum_reward = 0

            while not done:
                state = env.state
                action = self.policy(state, env.action_list)
                next_state, reward, done = env.step(action)
                sum_reward += reward
                self.q_state_init(next_state, env.action_list)

                # multi-step learning
                if len(self.multi_step_experience) == self.multi_step_window:
                    target_state = self.multi_step_experience[0].state
                    target_action = self.multi_step_experience[0].action
                    gain = 0
                    for t, experience in enumerate(self.multi_step_experience):
                        gain += discount_rate**t * experience.reward
                    gain += discount_rate**len(self.multi_step_experience) * \
                        max(self.Q[next_state].values())

                    td = gain - self.Q[target_state][target_action]
                    self.Q[target_state][target_action] += learning_rate * td

                self.multi_step_experience.append(
                    Experience(state, next_state, action, reward, done)
                )
                # end multi - step learning

                # td
                gain = reward + discount_rate * \
                    max(self.Q[next_state].values())
                td = gain - self.Q[state][action]
                self.Q[state][action] += learning_rate * td
                # end td

            sum_reward_list.append(sum_reward)

            if episode % 10 == 0:
                print("Episode ", episode)

        return sum_reward_list