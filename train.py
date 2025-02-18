# train_module.py
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

#####################################
# 1. 게임 환경 (강화학습 전용)
#####################################
class ExitStrategyEnv:
    def __init__(self):
        self.board_size = 7
        self.green_zone = (3, 3)  # 목표(초록) 구역
        self.obstacles = [(3, 0), (3, 6)]  # 장애물 위치
        self.reset()

    def reset(self):
        """게임 보드 초기화 및 상태 반환"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        for ox, oy in self.obstacles:
            self.board[ox, oy] = -2  # 장애물 표기
        self.scores = {1: 0, 2: 0}
        self.pieces = {1: [], 2: []}
        self.current_player = 1
        self.placement_phase = True  # 초기 말 배치 단계
        return self.get_state()

    def get_state(self):
        """현재 보드 상태 반환 (깊은 복사)"""
        return np.copy(self.board)

    def is_valid_placement(self, x, y):
        """해당 위치에 말을 배치할 수 있는지 판단"""
        if (x, y) in self.obstacles or (x, y) == self.green_zone:
            return False
        if self.current_player == 1 and y > 2:  # 플레이어 1: 왼쪽 영역
            return False
        if self.current_player == 2 and y < 4:  # 플레이어 2: 오른쪽 영역
            return False
        return self.board[x, y] == 0

    def place_piece(self, x, y):
        """말 배치 (배치 단계)"""
        if self.is_valid_placement(x, y):
            self.board[x, y] = self.current_player
            self.pieces[self.current_player].append((x, y))
            # 두 플레이어 모두 6개의 말을 배치하면 배치 단계 종료
            if len(self.pieces[1]) == 6 and len(self.pieces[2]) == 6:
                self.placement_phase = False
            self.current_player = 3 - self.current_player  # 턴 교체
            return 0, False  # 보상 0, 게임 종료 False
        return -1, False  # 잘못된 배치 시 -1 보상

    def get_valid_moves(self, x, y):
        """특정 말이 이동할 수 있는 모든 위치 반환 (상하좌우 이동)"""
        moves = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x, y
            while 0 <= nx + dx < self.board_size and 0 <= ny + dy < self.board_size:
                nx, ny = nx + dx, ny + dy
                if self.board[nx, ny] in [-2, 1, 2]:
                    break
                moves.append((nx, ny))
        return moves

    def move_piece(self, x, y, nx, ny):
        """배치 단계 종료 후 말 이동"""
        if (nx, ny) not in self.get_valid_moves(x, y):
            return -1, False  # 잘못된 이동
        self.board[x, y] = 0
        self.board[nx, ny] = self.current_player
        self.pieces[self.current_player].remove((x, y))
        self.pieces[self.current_player].append((nx, ny))
        # 목표 영역 도달 시 점수 획득
        if (nx, ny) == self.green_zone:
            self.scores[self.current_player] += 1
            if self.scores[self.current_player] == 2:
                return 2, True  # 승리: 보상 2, 게임 종료 True
        self.current_player = 3 - self.current_player
        return 0, False

    def step(self, action):
        """
        강화학습 에이전트의 행동 실행.
        action이 튜플 형태로 전달되며, 배치 단계에서는 (x,y),
        이동 단계에서는 (x, y, nx, ny) 형태입니다.
        """
        if self.placement_phase:
            return self.place_piece(*action)
        else:
            return self.move_piece(*action)

    def render(self):
        """현재 보드 출력 (디버깅 용)"""
        print(self.board)

#####################################
# 2. Q-Learning 에이전트
#####################################
class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9, q_table=None):
        self.epsilon = epsilon  # 탐험 확률
        self.alpha = alpha      # 학습률
        self.gamma = gamma      # 할인율
        self.q_table = q_table if q_table is not None else {}

    def get_q(self, state, action):
        """현재 상태, 행동에 대한 Q값 반환 (기본값 0.0)"""
        return self.q_table.get((tuple(state.flatten()), action), 0.0)

    def choose_action(self, env):
        """탐험/활용 방식으로 행동 선택"""
        if random.random() < self.epsilon:
            # 무작위 행동 선택 (탐험)
            if env.placement_phase:
                valid_actions = [(x, y) for x in range(env.board_size)
                                 for y in range(env.board_size)
                                 if env.is_valid_placement(x, y)]
                return random.choice(valid_actions) if valid_actions else None
            else:
                valid_actions = []
                for piece in env.pieces[env.current_player]:
                    moves = env.get_valid_moves(piece[0], piece[1])
                    for (nx, ny) in moves:
                        valid_actions.append((piece[0], piece[1], nx, ny))
                return random.choice(valid_actions) if valid_actions else None
        else:
            # Q-table 기반으로 최적 행동 선택 (활용)
            state = env.get_state()
            if env.placement_phase:
                valid_actions = [(x, y) for x in range(env.board_size)
                                 for y in range(env.board_size)
                                 if env.is_valid_placement(x, y)]
            else:
                valid_actions = []
                for piece in env.pieces[env.current_player]:
                    moves = env.get_valid_moves(piece[0], piece[1])
                    for (nx, ny) in moves:
                        valid_actions.append((piece[0], piece[1], nx, ny))
            if not valid_actions:
                return None
            return max(valid_actions, key=lambda a: self.get_q(state, a))

    def update_q(self, state, action, reward, next_state):
        """Q값 업데이트"""
        old_q = self.get_q(state, action)
        next_state_tuple = tuple(next_state.flatten())
        # 다음 상태에서 가능한 행동들 중 최대 Q값 산출
        possible_actions = [act for (s, act) in self.q_table.keys() if s == next_state_tuple]
        max_next_q = max([self.q_table[(next_state_tuple, a)] for a in possible_actions], default=0.0)
        self.q_table[(tuple(state.flatten()), action)] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)

    def get_best_action(self, state):
        """주어진 상태에서 Q-table 기반 최적 행동 반환"""
        state_tuple = tuple(state.flatten())
        actions = [action for (s, action) in self.q_table.keys() if s == state_tuple]
        if actions:
            return max(actions, key=lambda a: self.q_table[(state_tuple, a)])
        return None

#####################################
# 3. Q-table 저장/불러오기 함수
#####################################
def save_q_table(agent, filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(agent.q_table, f)

def load_q_table(filename="q_table.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

#####################################
# 4. 에이전트 학습 함수 (Self-Play)
#####################################
def train_agents(episodes=1000, agent1_epsilon=0.1, agent2_epsilon=0.1):
    env = ExitStrategyEnv()
    agent1 = QLearningAgent(epsilon=agent1_epsilon)
    agent2 = QLearningAgent(epsilon=agent2_epsilon)
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        # 한 에피소드 동안 게임 진행
        while not done:
            # 현재 턴의 에이전트 선택
            agent = agent1 if env.current_player == 1 else agent2
            action = agent.choose_action(env)
            if action is None:
                break
            reward, done = env.step(action)
            next_state = env.get_state()
            agent.update_q(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"에피소드 {episode + 1}/{episodes} 완료")

    return agent1, agent2, rewards

#####################################
# 5. 메인 실행: 에이전트 학습 및 결과 저장/시각화
#####################################
if __name__ == "__main__":
    # 에피소드 수 및 에이전트 하이퍼파라미터 설정
    EPISODES = 1000
    agent1, agent2, rewards = train_agents(episodes=EPISODES)

    # 학습된 Q-table 저장
    save_q_table(agent1, "agent1_q.pkl")
    save_q_table(agent2, "agent2_q.pkl")

    # 학습 성능 시각화
    plt.plot(rewards)
    plt.xlabel("에피소드")
    plt.ylabel("총 보상")
    plt.title("에이전트 학습 성능")
    plt.show()
