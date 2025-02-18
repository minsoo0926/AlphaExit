import random
import pickle
import numpy as np
from ExitStrategyEnv import ExitStrategyEnv

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q_table = {}
        self.epsilon = epsilon  # 탐색 확률
        self.alpha = alpha  # 학습률
        self.gamma = gamma  # 할인율

    def get_q(self, state, action):
        return self.q_table.get((self.state_representation(state), action), 0.0)

    def state_representation(self, state:ExitStrategyEnv):
        """상태를 각 플레이어의 말 위치, 턴, 배치 단계로 정의"""
        player_pieces = tuple(sorted(state.pieces[1])) + tuple(sorted(state.pieces[2]))
        turn = state.current_player
        placement_phase = state.placement_phase
        return (player_pieces, turn, placement_phase)

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
            actions = self.get_possible_actions(env)
            return max(actions, key=lambda a: self.get_q(self.state_representation(env), a), default=None)

    def update_q(self, state, action, reward, next_state):
        """Q 값 업데이트"""
        old_q = self.get_q(self.state_representation(state), action)
        next_state_representation = self.state_representation(next_state)
        max_next_q = max([self.get_q(next_state_representation, a) for a in self.get_possible_actions(next_state)], default=0.0)
        self.q_table[(self.state_representation(state), action)] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)

    def get_possible_actions(self, env):
        """가능한 모든 행동 반환"""
        if env.placement_phase:
            return [(x, y) for x in range(7) for y in range(7) if env.is_valid_placement(x, y)]
        else:
            pieces = env.pieces[env.current_player]
            return [(x, y, nx, ny) for x, y in pieces for nx, ny in env.get_valid_moves(x, y)]

    def save_q_table(self, filename="q_table.pkl"):
        """학습된 Q-table 저장"""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        """저장된 Q-table 불러오기"""
        with open(filename, "rb") as f:
            return pickle.load(f)

    def get_best_action(self, state):
        """Q-table을 기반으로 최적의 행동 선택"""
        state_representation = self.state_representation(state)
        actions = self.get_possible_actions(state)
        if actions:
            return max(actions, key=lambda a: self.get_q(state_representation, a))
        return None  # 가능한 행동이 없을 경우