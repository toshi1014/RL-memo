import numpy as np
import tensorflow as tf
from tensorflow import keras
from base import DiscreteAgent, Trainer


class ActorCritic(DiscreteAgent):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.epsilon = 0

    @classmethod
    def load(cls, model_path):
        agent = cls(0.)
        agent.model = keras.models.load_model(model_path)
        agent.action_space = list(range(agent.model.output[1].shape[1]))
        return agent

    def estimate(self, state):
        _, [action_prob] = self.model.predict(np.array([state]))
        return action_prob

    def make_model(self, feature_shape):
        input = keras.Input(shape=(feature_shape,))
        dense1 = keras.layers.Dense(
            units=1024,
            activation=tf.nn.relu,
        )(input)
        dense2 = keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
        )(dense1)
        v = keras.layers.Dense(units=1)(dense2)
        pi = keras.layers.Dense(
            units=len(self.action_space),
            activation=tf.nn.softmax,
        )(dense2)

        self.model = keras.Model(inputs=input, outputs=[v, pi])
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )

    def update(self, experience, discount_rate):
        with tf.GradientTape() as tape:
            [v_now], [action_prob] = self.model(
                np.array([experience.state]),
                training=False,
            )
            v_now = v_now[0]

            [v_next], _ = self.model(
                np.array([experience.next_state]),
                training=False,
            )
            v_next = v_next[0]

            one_hot_action = tf.one_hot(
                experience.action, len(self.action_space)
            )

            selected_action_prob = tf.reduce_sum(action_prob * one_hot_action)
            clipped = tf.clip_by_value(selected_action_prob, 1e-10, 1.)

            advantage = experience.reward + discount_rate * \
                v_next * (1 - int(experience.done)) - v_now
            actor_loss = - tf.math.log(clipped) * advantage
            critic_loss = advantage**2  # mse
            total_loss = actor_loss + critic_loss

        grad = tape.gradient(
            total_loss,
            self.model.trainable_variables,
        )
        self.model.optimizer.apply_gradients(
            zip(grad, self.model.trainable_variables)
        )


class ActorCriticTrainer(Trainer):
    def __init__(self, discount_rate):
        super().__init__(discount_rate)
        self.experience_list = []

    def train(self, env, agent, max_episodes):
        agent.action_space = list(range(env.action_space.n))
        reward_hist = super().train(env, agent, max_episodes)
        return agent, reward_hist, None

    def step(self, agent, experience):
        if not agent.model:
            agent.make_model(experience.state.shape[0])
        agent.update(experience, self.discount_rate)

    def episode_end(self, agent, episode):
        self.experience_list = []
