from environment import Environment, save_reward_map
from monte_carlo_agent import MonteCarloAgent
from q_learning_agent import QLearningAgent
from actor_critic_agent import Actor, Critic, ActorCriticAgent

epsilon = 0.1
max_episodes = 500
discount_rate = 0.9
learning_rate = 0.1
multi_step_window = 3

env = Environment()

monte_carlo_agent = MonteCarloAgent(epsilon)
reward_hist = monte_carlo_agent.learn(
    env, max_episodes, discount_rate
)

q_learning_agent = QLearningAgent(epsilon, multi_step_window)
reward_hist = q_learning_agent.learn(
    env, max_episodes, discount_rate, learning_rate
)

actor_critic_agent = ActorCriticAgent(Actor, Critic)
reward_hist = actor_critic_agent.learn(
    epsilon, env, max_episodes, discount_rate, learning_rate
)

save_reward_map(
    row_num=2,
    col_num=2,
    env=env,
    MonteCarlo=monte_carlo_agent.Q,
    QLearning=q_learning_agent.Q,
    ActorCritic=actor_critic_agent.actor.Q,
)