import random
import pickle

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q_table = {}
        self.epsilon = epsilon  # 탐색 확률
        self.alpha = alpha  # 학습률
        self.gamma = gamma  # 할인율

    def get_q(self, state, action):
        return self.q_table.get((tuple(state.flatten()), action), 0.0)

    def choose_action(self, env):
        """탐색/활용 기반 행동 선택"""
        if random.random() < self.epsilon:
            if env.placement_phase:
                return random.choice([(x, y) for x in range(7) for y in range(7) if env.is_valid_placement(x, y)])
            else:
                pieces = env.pieces[env.current_player]
                valid_moves = [(x, y, nx, ny) for x, y in pieces for nx, ny in env.get_valid_moves(x, y)]
                return random.choice(valid_moves) if valid_moves else None
        else:
            state = env.get_state()
            actions = [(x, y) for x in range(7) for y in range(7) if env.is_valid_placement(x, y)]
            if not env.placement_phase:
                pieces = env.pieces[env.current_player]
                actions = [(x, y, nx, ny) for x, y in pieces for nx, ny in env.get_valid_moves(x, y)]
            return max(actions, key=lambda a: self.get_q(state, a), default=None)

    def update_q(self, state, action, reward, next_state):
        """Q 값 업데이트"""
        old_q = self.get_q(state, action)
        max_next_q = max([self.get_q(next_state, a) for a in range(7)], default=0.0)
        self.q_table[(tuple(state.flatten()), action)] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)

    def save_q_table(self, filename="q_table.pkl"):
        """학습된 Q-table 저장"""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        """저장된 Q-table 불러오기"""
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)

    def get_best_action(self, state):
        """Q-table을 기반으로 최적의 행동 선택"""
        state_tuple = tuple(state.flatten())
        actions = [action for action in self.q_table.keys() if action[0] == state_tuple]
        if actions:
            return max(actions, key=lambda a: self.q_table[a])