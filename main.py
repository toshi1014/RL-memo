from environment import Environment, show_reward_map
from monte_carlo_agent import MonteCarloAgent
from q_learning_agent import QLearningAgent
from sarsa_agent import SARSAAgent
from actor_critic_agent import Actor, Critic, ActorCriticAgent


env = Environment()

monte_carlo_agent = MonteCarloAgent()
monte_carlo_agent.learn(env)

q_learning_agent = QLearningAgent()
q_learning_agent.learn(env)

sarsa_agent = SARSAAgent()
sarsa_agent.learn(env)

actor_critic_agent = ActorCriticAgent(Actor, Critic)
actor_critic_agent.learn(env)

show_reward_map(
    row_num=2,
    col_num=2,
    env=env,
    MonteCarlo=monte_carlo_agent.Q,
    QLearning=q_learning_agent.Q,
    SARSA=sarsa_agent.Q,
    ActorCritic=actor_critic_agent.actor.Q,
)