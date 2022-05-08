from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from base import DiscreteAgent, Trainer


class DDQN(DiscreteAgent):
    def __init__(self, learning_rate,  initial_epsilon=0, final_epsilon=0):
        super().__init__(learning_rate)
        self.initial_epsilon = initial_epsilon
        self.epsilon_diff = (self.initial_epsilon - final_epsilon)
        self.epsilon = self.initial_epsilon

    def make_model(self, feature_shape):
        input = keras.Input(shape=(feature_shape,))
        dense1 = keras.layers.Dense(
            units=10,
            activation=tf.nn.relu,
        )(input)
        dense2 = keras.layers.Dense(
            units=10,
            activation=tf.nn.relu,
        )(dense1)
        output = keras.layers.Dense(
            units=len(self.action_space),
            activation=tf.nn.softmax,
        )(dense2)

        self.model = keras.Model(inputs=input, outputs=output)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.mse,
        )
        self.target_model = keras.models.clone_model(self.model)

    def update(self, experience_list, discount_rate):
        state_list = np.array([e.state for e in experience_list])
        estimated_list = self.model.predict(state_list)

        for idx, e in enumerate(experience_list):
            discounted_reward = e.reward

            if not e.done:
                future_reward = np.max(
                    self.target_model.predict(np.array([e.next_state]))[0]
                )
                discounted_reward += discount_rate * future_reward

            estimated_list[idx][e.action] = discounted_reward

        loss = self.model.train_on_batch(state_list, estimated_list)
        return loss

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


class DDQNTrainer(Trainer):
    def __init__(self, discount_rate, buffer_size, batch_size, target_model_update_freq):
        super().__init__(discount_rate)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_model_update_freq = target_model_update_freq
        self.experience_list = deque(maxlen=self.buffer_size)
        self.loss_hist = []

    def train(self, env, agent, max_episodes):
        self.max_episodes = max_episodes
        agent.action_space = list(range(env.action_space.n))
        reward_hist = super().train(env, agent, max_episodes)
        return agent, reward_hist, self.loss_hist

    def episode_end(self, agent, episode):
        if not agent.model:
            if len(self.experience_list) == self.buffer_size:
                agent.make_model(self.experience_list[0].state.shape[0])

        else:
            batch = random.sample(self.experience_list, self.batch_size)
            loss_now = agent.update(batch, self.discount_rate)
            self.loss_hist.append(loss_now)

            if episode % self.target_model_update_freq == 0:
                agent.update_target_model()

        agent.epsilon = agent.initial_epsilon - \
            agent.epsilon_diff * (episode / self.max_episodes)