import random
import pickle
import os
from train_module import ExitStrategyEnv

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