from collections import namedtuple
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DEFAULT_REWARD = -0.04
GOAL = "o"
FAIL = "x"
BLOCKED = " "
GRID = [
    ["*", "*", "*", "*"],
    ["*", "*", "*", "x"],
    ["*", "*", "*", "*"],
    ["x", "*", "*", "o"],
]


Experience = namedtuple(
    "Experience",
    ["state", "next_state", "action", "reward", "done"],
)


class Action(Enum):
    UP = -1
    DOWN = 1
    LEFT = -2
    RIGHT = 2


class State:
    def __init__(self, row=0, col=0):
        self.row = row
        self.col = col

    def __repr__(self):
        return f"State: row {self.row}, col {self.col}"

    def __eq__(self, other):
        return [self.row, self.col] == [other.row, other.col]

    def __hash__(self):
        return hash((self.row, self.col))

    def clone(self):
        return State(self.row, self.col)


class Environment:
    def __init__(self, grid=GRID, default_reward=DEFAULT_REWARD):
        self.state = State()
        self.grid = grid
        self.default_reward = default_reward

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def col_length(self):
        return len(self.grid[0])

    @property
    def action_list(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    def reset(self):
        self.state = State()

    def move(self, state, action):
        next_state = state.clone()
        if action in[Action.UP, Action.DOWN]:
            next_state.row = np.clip(
                next_state.row + action.value, 0, self.row_length - 1
            )
        elif action in [Action.LEFT, Action.RIGHT]:
            next_state.col = np.clip(
                next_state.col + int(action.value / 2), 0, self.col_length - 1
            )
        else:
            raise ValueError("Unknown action ", action)

        if self.grid[next_state.row][next_state.col] == BLOCKED:
            next_state = state

        return next_state

    def reward_func(self, state, next_state):
        attr = self.grid[next_state.row][next_state.col]

        if attr == GOAL:
            reward = 1
            done = True
        elif attr == FAIL:
            reward = -1
            done = True
        else:
            reward = self.default_reward
            done = False

        return reward, done

    def transit_func(self, state, action):
        next_state = self.move(state, action)
        return {next_state: 1}

    def transit(self, state, action):
        transit_prob_dict = self.transit_func(state, action)
        next_state = np.random.choice(
            list(transit_prob_dict.keys()),
            p=list(transit_prob_dict.values()),
        )
        reward, done = self.reward_func(state, next_state)
        return next_state, reward, done

    def step(self, action):
        next_state, reward, done = self.transit(self.state, action)
        self.state = next_state
        return next_state, reward, done


def save_reward_map(row_num, col_num, env, **kwargs):
    state_size = 3
    q_row = env.row_length * state_size
    q_col = env.col_length * state_size

    fig = plt.figure()

    for i, title in enumerate(kwargs):
        Q = kwargs[title]
        reward_map = np.zeros((q_row, q_col))

        for r in range(env.row_length):
            for c in range(env.col_length):
                state = State(r, c)

                if not state in Q:
                    continue

                map_row = 1 + r * state_size
                map_col = 1 + c * state_size

                reward_map[map_row - 1][map_col] = Q[state][Action.UP]
                reward_map[map_row + 1][map_col] = Q[state][Action.DOWN]
                reward_map[map_row][map_col - 1] = Q[state][Action.LEFT]
                reward_map[map_row][map_col + 1] = Q[state][Action.RIGHT]
                reward_map[map_row][map_col] = np.mean(list(Q[state].values()))

        ax = fig.add_subplot(row_num, col_num, i + 1)
        ax.imshow(
            reward_map, cmap=cm.RdYlGn, interpolation="bilinear",
            vmax=abs(reward_map).max(), vmin=-abs(reward_map).max()
        )
        ax.set_xlim(-0.5, q_col - 0.5)
        ax.set_ylim(-0.5, q_row - 0.5)
        ax.set_xticks(np.arange(-0.5, q_col, state_size))
        ax.set_yticks(np.arange(-0.5, q_row, state_size))
        ax.set_xticklabels(range(env.col_length + 1))
        ax.set_yticklabels(range(env.row_length + 1))
        ax.set_title(title)
        ax.grid(which="both")
        plt.gca().invert_yaxis()

    plt.savefig(f"reward_maps.png")