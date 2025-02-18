# rl_agent.py
import random
import pickle
import numpy as np

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9, q_table=None):
        self.epsilon = epsilon  # 탐험 확률
        self.alpha = alpha      # 학습률
        self.gamma = gamma      # 할인율
        self.q_table = q_table if q_table is not None else {}

    def get_q(self, state, action):
        return self.q_table.get((tuple(state.flatten()), action), 0.0)

    def choose_action(self, env):
        if random.random() < self.epsilon:
            if env.placement_phase:
                valid_actions = [(x, y) for x in range(env.board_size)
                                 for y in range(env.board_size)
                                 if env.is_valid_placement(x, y)]
                return random.choice(valid_actions) if valid_actions else None
            else:
                valid_actions = []
                for piece in env.player_pieces[env.current_player]:
                    moves = env.get_valid_moves(piece.x, piece.y)
                    for (_, (nx, ny)) in moves:
                        valid_actions.append((piece.x, piece.y, nx, ny))
                return random.choice(valid_actions) if valid_actions else None
        else:
            state = env.get_state()
            if env.placement_phase:
                valid_actions = [(x, y) for x in range(env.board_size)
                                 for y in range(env.board_size)
                                 if env.is_valid_placement(x, y)]
            else:
                valid_actions = []
                for piece in env.player_pieces[env.current_player]:
                    moves = env.get_valid_moves(piece.x, piece.y)
                    for (_, (nx, ny)) in moves:
                        valid_actions.append((piece.x, piece.y, nx, ny))
            if not valid_actions:
                return None
            return max(valid_actions, key=lambda a: self.get_q(state, a))

    def update_q(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        next_state_tuple = tuple(next_state.flatten())
        possible_actions = [a for (s, a) in self.q_table.keys() if s == next_state_tuple]
        max_next_q = max([self.q_table[(next_state_tuple, a)] for a in possible_actions], default=0.0)
        self.q_table[(tuple(state.flatten()), action)] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)

    def get_best_action(self, state):
        state_tuple = tuple(state.flatten())
        actions = [action for (s, action) in self.q_table.keys() if s == state_tuple]
        if actions:
            return max(actions, key=lambda a: self.q_table[(state_tuple, a)])
        return None

def save_q_table(agent, filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(agent.q_table, f)

def load_q_table(filename="q_table.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)
