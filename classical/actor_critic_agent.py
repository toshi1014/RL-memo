import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from agent import Agent


class Actor(Agent):
    def __init__(self, epsilon, env):
        super().__init__(epsilon)
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    # not greedy
    def policy(self, state, action_list, greedy=None):
        self.q_state_init(state, action_list)
        val_list = np.array(list(self.Q[state].values()))
        return np.random.choice(action_list, p=self.softmax(val_list))


class Critic:
    def __init__(self):
        self.V = defaultdict(lambda: 0)


class ActorCriticAgent():
    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class

    def learn(self, epsilon, env, max_episodes, discount_rate, learning_rate):
        self.actor = self.actor_class(epsilon, env)
        self.critic = self.critic_class()

        sum_reward_list = []

        for episode in range(max_episodes):
            done = False
            env.reset()
            sum_reward = 0

            while not done:
                state = env.state
                action = self.actor.policy(state, env.action_list)
                next_state, reward, done = env.step(action)

                gain = reward + self.critic.V[next_state]
                td = gain - self.critic.V[state]

                self.actor.Q[state][action] += learning_rate * td
                self.critic.V[state] += learning_rate * td

                sum_reward += reward

            sum_reward_list.append(sum_reward)

            if episode % 10 == 0:
                print("Episode ", episode)

        return sum_reward_list