from environment import Environment, show_reward_map
from monte_carlo_agent import MonteCarloAgent
from q_learning_agent import QLearningAgent
from sarsa_agent import SARSAAgent


env = Environment()

monte_carlo_agent = MonteCarloAgent()
monte_carlo_agent.learn(env)

q_learning_agent = QLearningAgent()
q_learning_agent.learn(env)

sarsa_agent = SARSAAgent()
sarsa_agent.learn(env)

show_reward_map(
    row_num=1,
    col_num=3,
    env=env,
    MonteCarlo=monte_carlo_agent.Q,
    QLearning=q_learning_agent.Q,
    SARSA=sarsa_agent.Q,
)