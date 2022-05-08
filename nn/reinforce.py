import numpy as np
import tensorflow as tf
from tensorflow import keras
from base import DiscreteAgent, Trainer


class REINFORCE(DiscreteAgent):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.epsilon = 0

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
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )

    def update(self, state_list, action_space, reward_list):
        with tf.GradientTape() as tape:
            sum_loss = 0
            for idx, (state, action, reward) in enumerate(
                zip(state_list, action_space, reward_list)
            ):
                one_hot_action = tf.one_hot(action, len(self.action_space))
                action_prob = self.model(np.array([state]), training=False)[0]
                selected_action_prob = tf.reduce_sum(
                    action_prob * one_hot_action)
                clipped = tf.clip_by_value(selected_action_prob, 1e-10, 1.)
                sum_loss += - tf.math.log(clipped) * reward

        grad = tape.gradient(sum_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(grad, self.model.trainable_variables)
        )


class REINFORCETrainer(Trainer):
    def __init__(self, discount_rate):
        super().__init__(discount_rate)
        self.experience_list = []

    def train(self, env, agent, max_episodes):
        agent.action_space = list(range(env.action_space.n))
        reward_hist = super().train(env, agent, max_episodes)
        return agent, reward_hist, None

    def episode_end(self, agent, episode):
        state_list = [e.state for e in self.experience_list]
        action_list = [e.action for e in self.experience_list]
        discounted_reward_list = np.zeros((len(self.experience_list)))
        discounted_reward_list[-1] = self.experience_list[-1].reward

        if not agent.model:
            agent.make_model(self.experience_list[0].state.shape[0])

        for idx in reversed(range(len(self.experience_list) - 1)):
            discounted_reward_list[idx] = self.discount_rate * \
                discounted_reward_list[idx + 1] + \
                self.experience_list[idx].reward

        baseline = np.mean(discounted_reward_list)
        agent.update(
            state_list,
            action_list,
            discounted_reward_list - baseline
        )
        self.experience_list = []
