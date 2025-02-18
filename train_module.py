# train_module.py
import os
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

plt.ion()  # 인터랙티브 모드 활성화 (실시간 시각화)

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
        """게임 보드를 초기화하고 초기 상태 반환"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        for ox, oy in self.obstacles:
            self.board[ox, oy] = -2  # 장애물 표기
        self.scores = {1: 0, 2: 0}
        self.pieces = {1: [], 2: []}
        self.current_player = 1
        self.placement_phase = True  # 초기 말 배치 단계
        self.winner = None
        return self.get_state()

    def get_state(self):
        """현재 보드 상태를 깊은 복사해서 반환"""
        return np.copy(self.board)

    def is_valid_placement(self, x, y):
        """해당 위치에 말을 배치할 수 있는지 판단"""
        if (x, y) in self.obstacles or (x, y) == self.green_zone:
            return False
        if self.current_player == 1 and y > 2:
            return False
        if self.current_player == 2 and y < 4:
            return False
        return self.board[x, y] == 0
    
    def sort_pieces(self):
        """각 플레이어의 말 목록을 (x, y) 오름차순으로 정렬합니다."""
        for player in self.pieces:
            self.pieces[player].sort(key=lambda pos: (pos[0], pos[1]))

    def place_piece(self, x, y):
        """배치 단계에서 말 배치; 올바른 배치면 보상 0, 틀리면 -1을 반환"""
        if self.is_valid_placement(x, y):
            self.board[x, y] = self.current_player
            self.pieces[self.current_player].append((x, y))
            self.sort_pieces()  # 말 목록 정렬
            if len(self.pieces[1]) == 6 and len(self.pieces[2]) == 6:
                self.placement_phase = False
            self.current_player = 3 - self.current_player  # 턴 교체
            return 0, False
        return -1, False

    def get_valid_moves(self, x, y):
        """각 방향에서 끝까지 이동한 최종 위치만 반환합니다."""
        moves = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            curr_x, curr_y = x, y
            while True:
                next_x = curr_x + dx
                next_y = curr_y + dy
                if not (0 <= next_x < 7 and 0 <= next_y < 7):
                    break
                if self.board[next_x, next_y] in [-2, 1, 2]:
                    break
                curr_x, curr_y = next_x, next_y
            if (curr_x, curr_y) != (x, y):
                moves.append((curr_x, curr_y))
        return moves

    def is_gray_zone(self, x, y):
        """
        green zone과 장애물을 제외하고, 중앙 행 또는 열에 위치한 곳을 gray zone으로 정의합니다.
        """
        if (x, y) == self.green_zone or (x, y) in self.obstacles:
            return False
        return x == 3 or y == 3
    
    def move_piece(self, x, y, nx, ny):
        """
        이동 단계에서 말 이동.
        목적지로 이동한 후, green zone이면 +100, gray zone이면 +10의 보상을 부여합니다.
        유효하지 않은 행동은 -1의 보상을 반환합니다.
        """
        if (nx, ny) not in self.get_valid_moves(x, y):
            return -1, False
        self.board[x, y] = 0
        self.board[nx, ny] = self.current_player
        self.pieces[self.current_player].remove((x, y))
        self.pieces[self.current_player].append((nx, ny))
        self.sort_pieces()  # 정렬 수행
        reward = 0
        if (nx, ny) == self.green_zone:
            self.scores[self.current_player] += 1
            reward = 100
            if self.scores[self.current_player] == 2:
                self.winner = self.current_player
                return reward, True
        
        self.current_player = 3 - self.current_player
        return reward, False

    def step(self, action):
        """
        강화학습 에이전트의 행동 실행.
        배치 단계에서는 action이 (x, y),
        이동 단계에서는 action이 (x, y, nx, ny) 형태입니다.
        턴 종료 시, 행동한 플레이어가 gray zone 위에 유지 중인 말 개수당 1점씩 추가 보상을 부여합니다.
        """
        # 현재 행동을 수행하는 플레이어 저장
        acting_player = self.current_player
        if self.placement_phase:
            reward, done = self.place_piece(*action)
        else:
            reward, done = self.move_piece(*action)
        
        # 행동한 플레이어가 gray zone 위에 유지 중인 말 개수만큼 추가 보상
        maintenance_reward = sum(10 for piece in self.pieces[acting_player]
                                  if self.is_gray_zone(piece[0], piece[1]))
        total_reward = reward + maintenance_reward
        return total_reward, done


    def render(self):
        """현재 보드 상태 출력 (디버깅 용)"""
        print(self.board)

#####################################
# 2. Q-Learning 에이전트 (플레이어 인식 포함)
#####################################
class QLearningAgent:
    def __init__(self, player, epsilon=0.1, alpha=0.5, gamma=0.9, q_table=None):
        self.player = player  # 에이전트 담당 플레이어 (1 또는 2)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = q_table if q_table is not None else {}

    def get_q(self, state, action):
        return self.q_table.get((tuple(state.flatten()), self.player, action), 0.0)

    def choose_action(self, env):
        state = env.get_state()
        if random.random() < self.epsilon:
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
            action = random.choice(valid_actions) if valid_actions else None
        else:
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
                action = None
            else:
                q_vals = [self.get_q(state, a) for a in valid_actions]
                if max(q_vals) == 0:
                    action = random.choice(valid_actions)
                else:
                    action = max(valid_actions, key=lambda a: self.get_q(state, a))
        return action

    def update_q(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        next_state_tuple = tuple(next_state.flatten())
        possible_actions = [a for (s, p, a) in self.q_table.keys() if s == next_state_tuple and p == self.player]
        max_next_q = max([self.q_table.get((next_state_tuple, self.player, a), 0.0) for a in possible_actions], default=0.0)
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(tuple(state.flatten()), self.player, action)] = new_q

    def get_best_action(self, state):
        state_tuple = tuple(state.flatten())
        actions = [action for (s, p, action) in self.q_table.keys() if s == state_tuple and p == self.player]
        if actions:
            return max(actions, key=lambda a: self.q_table[(state_tuple, self.player, a)])
        return None

#####################################
# 3. Q-table 저장/불러오기 함수
#####################################
def save_q_table(agent, filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(agent.q_table, f)

def load_q_table(filename="q_table.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"{filename} 파일을 찾을 수 없습니다. 빈 Q-table로 시작합니다.")
        return {}

#####################################
# 4. 에이전트 학습 함수 (Self-Play) 및 시각화
#####################################
def train_agents(episodes=1000, agent1_epsilon=0.1, agent2_epsilon=0.1):
    env = ExitStrategyEnv()

    # 에이전트 생성 (기존 파일이 있으면 로드)
    if os.path.exists("agent1_q.pkl"):
        q_table1 = load_q_table("agent1_q.pkl")
        agent1 = QLearningAgent(player=1, epsilon=agent1_epsilon, q_table=q_table1)
        print("Agent1의 Q-table 로드됨.")
    else:
        agent1 = QLearningAgent(player=1, epsilon=agent1_epsilon)
        print("새로운 Agent1 생성됨.")

    if os.path.exists("agent2_q.pkl"):
        q_table2 = load_q_table("agent2_q.pkl")
        agent2 = QLearningAgent(player=2, epsilon=agent2_epsilon, q_table=q_table2)
        print("Agent2의 Q-table 로드됨.")
    else:
        agent2 = QLearningAgent(player=2, epsilon=agent2_epsilon)
        print("새로운 Agent2 생성됨.")

    rewards = []
    fig, ax = plt.subplots()

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        # 에피소드 당 내부 출력은 최소화하고 중요한 정보만 출력
        while not done:
            current_player = env.current_player
            agent = agent1 if current_player == 1 else agent2
            action = agent.choose_action(env)
            if action is None:
                break
            reward, done = env.step(action)
            next_state = env.get_state()
            agent.update_q(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        # 100 에피소드마다 간단한 요약 출력
        if (episode + 1) % 100 == 0 or episode == 0:
            winner = env.winner if env.winner is not None else "정보 없음"
            print(f"에피소드 {episode+1}: 총 보상 = {total_reward}, 승자 = {winner}")
            ax.clear()
            ax.imshow(env.get_state(), cmap="viridis")
            ax.set_title(f"에피소드 {episode+1} 최종 보드 상태")
            plt.pause(0.5)

    plt.ioff()
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("에피소드")
    plt.ylabel("총 보상")
    plt.title("에이전트 학습 성능")
    plt.show()

    return agent1, agent2, rewards

#####################################
# 5. 메인 실행: 에이전트 학습 및 결과 저장
#####################################
if __name__ == "__main__":
    EPISODES = 1000
    agent1, agent2, rewards = train_agents(episodes=EPISODES)
    
    save_q_table(agent1, "agent1_q.pkl")
    save_q_table(agent2, "agent2_q.pkl")
