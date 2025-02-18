import numpy as np
import random

class ExitStrategyEnv:
    def __init__(self):
        self.board_size = 7
        self.green_zone = (3, 3)
        self.obstacles = [(3, 0), (3, 6)]
        self.players = [1, 2]
        self.reset()

    def reset(self):
        """게임 보드를 초기화하고 초기 상태 반환"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        for ox, oy in self.obstacles:
            self.board[ox, oy] = -2  # 장애물
        self.scores = {1: 0, 2: 0}
        self.pieces = {1: [], 2: []}
        self.current_player = 1
        self.placement_phase = True
        return self.get_state()

    def get_state(self):
        """현재 게임 상태 반환"""
        return np.copy(self.board)

    def is_valid_placement(self, x, y):
        """말을 배치할 수 있는지 검사"""
        if (x, y) in self.obstacles or (x, y) == self.green_zone:
            return False
        if self.current_player == 1 and y > 2:
            return False
        if self.current_player == 2 and y < 4:
            return False
        return self.board[x, y] == 0

    def place_piece(self, x, y):
        """말 배치"""
        if self.is_valid_placement(x, y):
            self.board[x, y] = self.current_player
            self.pieces[self.current_player].append((x, y))
            if len(self.pieces[1]) == 6 and len(self.pieces[2]) == 6:
                self.placement_phase = False  # 배치 완료
            self.current_player = 3 - self.current_player
            return 0, False
        return -1, False  # 잘못된 배치

    def get_valid_moves(self, x, y):
        """해당 말이 이동할 수 있는 위치 반환"""
        moves = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x, y
            while 0 <= nx + dx < 7 and 0 <= ny + dy < 7:
                nx, ny = nx + dx, ny + dy
                if self.board[nx, ny] in [-2, 1, 2]:
                    break
                moves.append((nx, ny))
        return moves

    def move_piece(self, x, y, nx, ny):
        """말 이동"""
        if (nx, ny) not in self.get_valid_moves(x, y):
            return -1, False  # 잘못된 이동
        
        self.board[x, y] = 0
        self.board[nx, ny] = self.current_player
        self.pieces[self.current_player].remove((x, y))
        self.pieces[self.current_player].append((nx, ny))

        if (nx, ny) == self.green_zone:
            self.scores[self.current_player] += 1
            if self.scores[self.current_player] == 2:
                return 2, True  # 승리
        self.current_player = 3 - self.current_player
        return 0, False  # 정상 이동

    def step(self, action):
        """강화학습 에이전트가 실행할 행동"""
        if self.placement_phase:
            return self.place_piece(*action)
        else:
            return self.move_piece(*action)

    def render(self):
        """게임 보드 출력"""
        print(self.board)
