# train_module.py
import os
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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
        board_flat = np.copy(self.board).flatten()
        phase = 1 if self.placement_phase else 0
        return np.append(board_flat, [self.current_player, phase])
    
    
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

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 3. DQN 에이전트 (Q-LearningAgent 대체)
class DQNAgent:
    def __init__(self, player, epsilon=0.1, lr=0.001, gamma=0.9):
        self.player = player
        self.epsilon = epsilon
        self.gamma = gamma
        self.input_dim = 51 + 4  # 상태(51) + 액션 특성(4)
        self.model = DQN(self.input_dim).to(device)
        self.target_model = DQN(self.input_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.step_count = 0
        self.last_experience = None  # 마지막 경험 저장

    def get_action_features(self, action):
        if len(action) == 2:  # 배치 액션
            return [action[0], action[1], 0, 0]
        return list(action[:4])  # 이동 액션

    def get_state_action(self, state, action):
        action_features = self.get_action_features(action)
        return np.concatenate([state, action_features])

    def choose_action(self, env):
        state = env.get_state()
        valid_actions = self.get_valid_actions(env)
        if not valid_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # 신경망을 통해 최적 액션 선택
        q_values = []
        for action in valid_actions:
            sa = self.get_state_action(state, action)
            sa_tensor = torch.FloatTensor(sa).unsqueeze(0).to(device)  # 데이터를 GPU로 이동
            with torch.no_grad():
                q_values.append(self.model(sa_tensor).item())
        return valid_actions[np.argmax(q_values)]

    def get_valid_actions(self, env):
        if env.placement_phase:
            return [(x, y) for x in range(7) for y in range(7) 
                    if env.is_valid_placement(x, y)]
        else:
            actions = []
            for (x, y) in env.pieces[env.current_player]:
                moves = env.get_valid_moves(x, y)
                actions.extend([(x, y, nx, ny) for (nx, ny) in moves])
            return actions

    def store_experience(self, state, action, reward):
        self.last_experience = (state, action, reward)

    def update(self, curr_state, curr_action, opponent_reward, done):
        if self.last_experience is None:
            return
        
        state, action, reward = self.last_experience
        if opponent_reward >= 100:
            reward -= opponent_reward  # 상대가 득점하면 페널티 부여

        target_q = reward if done else reward + self.gamma * self.get_q(curr_state, curr_action)

        sa = self.get_state_action(state, action)
        sa_tensor = torch.FloatTensor(sa).unsqueeze(0).to(device)
        target_q_tensor = torch.tensor([[target_q]], device=device, dtype=torch.float32)

        self.optimizer.zero_grad()
        loss = F.mse_loss(self.model(sa_tensor), target_q_tensor)
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % 100 == 0:
            print("target model updated")
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.last_experience = None  # 경험 초기화

    def get_max_q(self, env, state):
        valid_actions = self.get_valid_actions(env)
        if not valid_actions:
            return 0.0
        q_values = [self.model(torch.FloatTensor(self.get_state_action(state, a)).to(device)).item() for a in valid_actions]
        return max(q_values)

    def get_q(self, state, action):
        if not action:
            return 0.0
        q_value = self.model(torch.FloatTensor(self.get_state_action(state, action)).to(device)).item()
        return q_value

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
# 4. 학습 함수 수정
def train_agents_with_dqn(episodes=1000):
    env = ExitStrategyEnv()
    agent1 = DQNAgent(1)
    agent2 = DQNAgent(2)

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            for agent in [agent1, agent2]:
                state = env.get_state()
                action = agent.choose_action(env)
                if action is None:
                    continue
                reward, done = env.step(action)
                agent.store_experience(state, action, reward)
                
                opponent_agent = agent2 if agent == agent1 else agent1
                if opponent_agent.last_experience is not None:
                    opponent_state, opponent_action, opponent_reward = opponent_agent.last_experience
                else:
                    opponent_reward = 0
                    
                agent.update(state, action, opponent_reward, done)
        
        print("model updated")
        torch.save(agent1.model.state_dict(), "agent1_dqn.pth")
        torch.save(agent2.model.state_dict(), "agent2_dqn.pth")

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
    last_actions = {1: None, 2: None}

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
            opponent = 3 - current_player

            if reward >= 100 and last_actions[opponent] is not None:
                opp_state, opp_action = last_actions[opponent]
                opponent_agent = agent1 if opponent == 1 else agent2
                penalty = -reward  # 득점 보상과 동일한 크기의 페널티
                opponent_agent.update_q(opp_state, opp_action, penalty, next_state)
                last_actions[opponent] = None

            last_actions[current_player] = (state, action)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        # 100 에피소드마다 간단한 요약 출력
        if (episode + 1) % 100 == 0 or episode == 0:
            winner = env.winner if env.winner is not None else "정보 없음"
            print(f"에피소드 {episode+1}: 총 보상 = {total_reward}, 승자 = {winner}")

    return agent1, agent2, rewards


def train_agents_with_opponent_update(episodes=1000, agent1_epsilon=0.1, agent2_epsilon=0.1):
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
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        # 각 플레이어의 마지막 state-action-reward를 저장할 딕셔너리 (키: 플레이어 번호)
        last_state_action = {}

        while not done:
            current_player = env.current_player
            agent_current = agent1 if current_player == 1 else agent2
            state_current = env.get_state()
            action_current = agent_current.choose_action(env)
            if action_current is None:
                done = True
                break
            reward, done = env.step(action_current)
            total_reward += reward
            new_state = env.get_state()  # 현재 행동에 의해 변경된 상태

            # 상대방의 이전 행동이 있다면, 새 상태(new_state)를 기반으로 Q 업데이트 진행
            opponent = 3 - current_player
            if opponent in last_state_action:
                s_last, a_last, r_last = last_state_action[opponent]
                opponent_agent = agent1 if opponent == 1 else agent2
                r_last = r_last - reward if reward >= 100 else r_last
                opponent_agent.update_q(s_last, a_last, r_last, new_state)
                # 업데이트 후 상대방의 메모리 초기화
                del last_state_action[opponent]

            # 현재 플레이어의 state-action-reward를 메모리에 저장 (나중에 상대 행동 후 업데이트)
            last_state_action[current_player] = (state_current, action_current, reward)

        # 게임 종료 후, 아직 업데이트되지 않은 플레이어의 경험에 대해 최종 상태를 사용하여 업데이트
        final_state = env.get_state()
        for player, (s, a, r) in last_state_action.items():
            agent = agent1 if player == 1 else agent2
            agent.update_q(s, a, r, final_state)
        last_state_action = {}

        rewards.append(total_reward)
        if (episode + 1) % 100 == 0 or episode == 0:
            winner = env.winner if env.winner is not None else "정보 없음"
            print(f"에피소드 {episode+1}: 총 보상 = {total_reward}, 승자 = {winner}")

    return agent1, agent2, rewards

#####################################
# 5. 메인 실행: 에이전트 학습 및 결과 저장
#####################################

if __name__ == "__main__":
    EPISODES = 100
    for i in range(EPISODES):
        train_agents_with_dqn(episodes=1)
