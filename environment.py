from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

MOVE_PROB = 1
DEFAULT_REWARD = -0.04
GOAL = "o"
FAIL = "x"
BLOCKED = " "
GRID = [
    ["*", "*", "*", "*"],
    ["*", "x", "*", "x"],
    ["*", "*", "*", "x"],
    ["x", "*", "*", "o"],
]


class State():
    def __init__(self, row_=0, col_=0):
        self.row = row_
        self.col = col_

    def __repr__(self):
        return f"State: row {self.row}, col {self.col}"

    def __eq__(self, other):
        return [self.row, self.col] == [other.row, other.col]

    def __hash__(self):
        return hash((self.row, self.col))

    def clone(self):
        return State(self.row, self.col)


class Action(Enum):
    UP = -1
    DOWN = 1
    LEFT = -2
    RIGHT = 2


class Environment():
    def __init__(self, grid_=GRID, move_prob_=MOVE_PROB, default_reward_=DEFAULT_REWARD):
        self.state = State()
        self.grid = grid_
        self.move_prob = MOVE_PROB
        self.default_reward = DEFAULT_REWARD

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def action_list(self):
        return [action for action in Action]


    def reset_state(self):
        self.state = State()


    def move(self, state, action):
        next_state = state.clone()

        if action in [Action.UP, Action.DOWN]:
            next_state.row += action.value
            next_state.row = np.clip(next_state.row, 0, self.row_length-1)

        elif action in [Action.LEFT, Action.RIGHT]:
            next_state.col += int(action.value/2)
            next_state.col = np.clip(next_state.col, 0, self.column_length-1)
        else:
            raise ValueError

        if self.grid[next_state.row][next_state.col] == BLOCKED:
            next_state = state

        return next_state


    def transit_func(self, state, action):
        transition_prob_dict = {}

        opposite_direction = Action(action.value * -1)

        for action_now in self.action_list:
            if action == action_now:
                prob = self.move_prob
            else:
                if action_now == opposite_direction:
                    prob = 0
                else:
                    prob = (1-self.move_prob)/2

            next_state = self.move(state, action_now)

            if next_state in transition_prob_dict:
                transition_prob_dict[next_state] += prob
            else:
                transition_prob_dict[next_state] = prob

        return transition_prob_dict


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


    def transit(self, state, action):
        transition_prob_dict = self.transit_func(state, action)
        if len(transition_prob_dict) == 0:
            return state, 0, False

        next_state_list = list(transition_prob_dict.keys())
        prob_list = [transition_prob_dict[next_state] for next_state in next_state_list]

        next_state = np.random.choice(next_state_list, p=prob_list)
        reward, done = self.reward_func(state, next_state)

        return next_state, reward, done


    def step(self, action):
        next_state, reward, done = self.transit(self.state, action)
        self.state = next_state
        return next_state, reward, done


def show_reward_map(row_num, col_num, env, **kwargs):
    state_size = 3
    q_row = env.row_length * state_size
    q_col = env.column_length * state_size

    fig = plt.figure()

    for i, title in enumerate(kwargs):
        Q = kwargs[title]
        reward_map = np.zeros((q_row, q_col))

        for r in range(env.row_length):
            for c in range(env.column_length):
                state = State(r, c)

                if not state in Q:
                    continue

                map_row = 1 + r * state_size
                map_col = 1 + c * state_size

                reward_map[map_row-1][map_col] = Q[state][0]
                reward_map[map_row+1][map_col] = Q[state][1]
                reward_map[map_row][map_col-1] = Q[state][2]
                reward_map[map_row][map_col+1] = Q[state][3]
                reward_map[map_row][map_col] = np.mean(Q[state])

        ax = fig.add_subplot(row_num, col_num, i+1)
        ax.imshow(
            reward_map, cmap=cm.RdYlGn, interpolation="bilinear",
            vmax=abs(reward_map).max(), vmin=-abs(reward_map).max()
        )
        ax.set_xlim(-0.5, q_col-0.5)
        ax.set_ylim(-0.5, q_row-0.5)
        ax.set_xticks(np.arange(-0.5, q_col, state_size))
        ax.set_yticks(np.arange(-0.5, q_row, state_size))
        ax.set_xticklabels(range(env.column_length + 1))
        ax.set_yticklabels(range(env.row_length + 1))
        ax.set_title(title)
        ax.grid(which="both")
        plt.gca().invert_yaxis()

    plt.show()