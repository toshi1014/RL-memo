from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from environment import Action, Environment, show_reward_map

MAX_EPISODE = 1000
GAMMA = 0.9
LEARNING_RATE = 0.1


class Actor():
    def __init__(self, env):
        default_prob = 1 / (env.row_length*env.column_length)
        self.Q = defaultdict(lambda: defaultdict(lambda: default_prob))

    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=0)

    def policy(self, state, arg_action_list):
        action_list = []
        value_list = []

        if not self.Q[state]:
            for action in arg_action_list:
                self.Q[state][action]

        for action in self.Q[state]:
            action_list.append(action)
            value_list.append(self.Q[state][action])

        return np.random.choice(action_list, p=self.softmax(np.array(value_list)))


class Critic():
    def __init__(self, env):
        self.V = defaultdict(lambda: 0)


class ActorCriticAgent():
    def __init__(self, actor_class_, critic_class_):
        self.actor_class = actor_class_
        self.critic_class = critic_class_

    def learn(self, env, max_episode=MAX_EPISODE, gamma=GAMMA, learning_rate=LEARNING_RATE):
        self.actor = self.actor_class(env)
        self.critic = self.critic_class(env)
        reward_list = []

        for episode in range(max_episode):
            env.reset_state()
            done = False
            sum_reward = 0

            while not done:
                state_now = env.state
                action = self.actor.policy(env.state, env.action_list)
                next_state, reward, done = env.step(action)

                gain = reward + gamma*self.critic.V[next_state]
                td = gain - self.critic.V[state_now]

                self.actor.Q[state_now][action] += learning_rate * td
                self.critic.V[state_now] += learning_rate * td

                sum_reward += reward

            reward_list.append(sum_reward)

            if episode % 100 == 0:
                print("episode", episode)

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.scatter(range(max_episode), reward_list)
        plt.show()


def train():
    actor_critic_agent = ActorCriticAgent(Actor, Critic)
    env = Environment()
    actor_critic_agent.learn(env)
    show_reward_map(
        row_num=1,
        col_num=2,
        env=env,
        ActorCriticQ=actor_critic_agent.actor.Q,
    )


if __name__ == '__main__':
    train()