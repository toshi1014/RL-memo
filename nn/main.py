import sys
import gym
import matplotlib.pyplot as plt
from ddqn import DDQN, DDQNTrainer
from reinforce import REINFORCE, REINFORCETrainer
from actor_critic import ActorCritic, ActorCriticTrainer
from ddpg import DDPG, DDPGTrainer


# params
discount_rate = 0.99
learning_rate = 0.01
actor_learning_rate = 0.00005
critic_learning_rate = 0.0005
buffer_size = 1024
batch_size = 64
target_model_update_freq = 1
initial_epsilon = 0.3
final_epsilon = 1e-3
noise_stddev = 0.1
target_trans_rate = 0.005
model_path = "model.h5"

max_episodes = 60

# env = gym.make("CartPole-v1")
# agent_type = "actor_critic"

env = gym.make("Pendulum-v1")
agent_type = "ddpg"

# end params


trainer_agent_dict = {
    "ddqn": [
        DDQNTrainer(
            discount_rate=discount_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_model_update_freq=target_model_update_freq,
        ),
        DDQN(
            initial_epsilon=initial_epsilon,
            final_epsilon=final_epsilon,
            learning_rate=learning_rate,
        ),
    ],
    "reinforce": [
        REINFORCETrainer(discount_rate=discount_rate),
        REINFORCE(learning_rate=learning_rate),
    ],
    "actor_critic": [
        ActorCriticTrainer(discount_rate=discount_rate),
        ActorCritic(learning_rate=learning_rate),
    ],
    "ddpg": [
        DDPGTrainer(
            discount_rate=discount_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
        ),
        DDPG(
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            noise_stddev=noise_stddev,
            target_trans_rate=target_trans_rate,
        ),
    ]
}


trainer, agent = trainer_agent_dict[agent_type]

if len(sys.argv) == 1:
    trained_agent, reward_hist, loss_hist = trainer.train(
        env, agent, max_episodes
    )
    trained_agent.save(model_path)
    plt.plot(range(len(reward_hist)), reward_hist)
    plt.title("reward_history - " + agent_type)
    plt.savefig("reward_history.png")
    plt.clf()

    if loss_hist != None:
        plt.plot(range(len(loss_hist)), loss_hist)
        plt.title("loss_history - " + agent_type)
        plt.savefig("loss_history.png")

else:
    agent = (agent.__class__).load(model_path)
    agent.play(env, 5)
