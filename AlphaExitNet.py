import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math, random, copy
import os
import numpy as np
import copy

MAX_TURNS = 200

class ExitStrategyEnv:
    def __init__(self, max_turns=MAX_TURNS):
        """
        board_size: 보드 크기 (7x7)
        required_placements: placement phase에서 각 플레이어가 말을 놓을 횟수 (여기서는 1)
        max_turns: 최대 턴 수 (초과 시 무승부 처리)
        """
        self.board_size = 7
        self.required_placements = 6
        self.max_turns = max_turns
        self.reset()
        
    def reset(self):
        # 상대 시각: 항상 현재 플레이어의 말이 채널 0에, 상대 말이 채널 1에 위치하도록 함.
        self.board = np.zeros((2, self.board_size, self.board_size), dtype=np.int32)
        self.board[0,3,0] =2; self.board[0,3,6]=2; self.board[1,3,0] =2; self.board[1,3,6]=2;
        self.phase = "placement"  # placement_phase부터 시작
        self.turn = 0
        self.current_player = 1  # 예: 1 또는 -1
        self.placements = {1: 0, -1: 0}  # 각 플레이어의 배치 횟수
        self.scores = {1: 0, -1: 0}       # 각 플레이어의 점수
        self.column_count = {0:0, 1:0, 2:0, 4:0, 5:0, 6:0}
        self.winner = 0
        return self.get_state()
    
    def get_state(self):
        """현재 상태를 dictionary 형태로 반환"""
        return {
            "board": self.board.copy(),
            "phase": self.phase,
            "turn": self.turn,
            "current_player": self.current_player,
            "placements": self.placements.copy(),
            "scores": self.scores.copy(),
            "column_count": self.column_count.copy(),
            "winner": self.winner
        }
        
    def get_legal_moves_placement(self):
        legal_moves = []
        for action in range(self.board_size * self.board_size):
            i = action // self.board_size
            j = action % self.board_size
            # 칸이 비어있고, 중앙 행/열 제외, 열 제한, 플레이어별 영역 조건을 만족해야 함
            if self.board[0, i, j] == 0 and self.board[1, i, j] == 0 and (3 not in [i, j]):
                if self.column_count[j] < 2:
                    if (self.current_player == 1 and j < 3) or (self.current_player == -1 and j > 3):
                        legal_moves.append(action)
        return legal_moves

    def get_legal_moves_movement(self):
        legal_moves = []
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        board_size = self.board_size
        positions = np.argwhere(self.board[0] == 1)
        positions = sorted(positions, key=lambda pos: (pos[0], pos[1]))
        for piece_index, (i, j) in enumerate(positions):
            for direction_index, (di, dj) in enumerate(directions):
                new_i, new_j = i, j
                while True:
                    next_i = new_i + di
                    next_j = new_j + dj
                    if not (0 <= next_i < board_size and 0 <= next_j < board_size):
                        break
                    if self.board[0, next_i, next_j] != 0 or self.board[1, next_i, next_j] != 0:
                        break
                    new_i, new_j = next_i, next_j
                if (new_i, new_j) != (i, j):
                    legal_moves.append(piece_index * 4 + direction_index)
        return legal_moves
    
    def step(self, action):
        """
        action:
          - placement_phase: 0 ~ (board_size*board_size - 1) 중 하나 (말을 놓을 위치)
          - movement_phase: 0 ~ 23 (6개 말 × 4방향) 중 하나 (어느 말과 어느 방향으로 이동)
        """
        if self.phase == "placement":
            return self._placement_step(action)
        elif self.phase == "movement":
            return self._movement_step(action)
        else:
            raise ValueError("Unknown phase: " + self.phase)
    
    def _placement_step(self, action):
        # 먼저 합법 행동 목록을 구함
        penalty = 0.0
        legal_moves = self.get_legal_moves_placement()
        if action not in legal_moves:
            # 불법 행동에 대해 페널티를 주고, resampling하여 합법 행동을 선택
            penalty = -0.5  # 불법 행동에 대한 페널티 값 (원하는 값으로 조정)
            if legal_moves:
                action = random.choice(legal_moves)
            else:
                # 합법 행동이 하나도 없으면 현재 상태를 그대로 반환하면서 에피소드 종료
                return self.get_state(), -1, True, {"error": "No legal placement moves available"}
        
        i = action // self.board_size
        j = action % self.board_size

        self.board[0, i, j] = 1
        self.placements[self.current_player] += 1
        self.column_count[j] += 1
        self.turn += 1
        
        # 페널티가 적용된 경우 턴마다 추가 보상을 줄 수도 있음 (여기서는 단순 출력)
        if penalty:
            pass  # 필요 시 추가 로직 구현 가능
        
        self._swap_board()
        
        if self.placements[1] >= self.required_placements and self.placements[-1] >= self.required_placements:
            self.phase = "movement"
        
        return self.get_state(), penalty, False, {}
    
    def _movement_step(self, action):
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        board_size = self.board_size
        legal_moves = self.get_legal_moves_movement()
        if action not in legal_moves:
            penalty = -0.5  # 불법 행동에 대한 페널티
            if legal_moves:
                action = random.choice(legal_moves)
            else:
                # 움직일 수 있는 말이 하나도 없다면 무승부 등 처리할 수 있음
                return self.get_state(), penalty, True, {"error": "No legal movement moves available"}
        else:
            penalty = 0

        positions = np.argwhere(self.board[0] == 1)
        positions = sorted(positions, key=lambda pos: (pos[0], pos[1]))
        
        piece_index = action // 4
        direction_index = action % 4
        if piece_index >= len(positions):
            # 이 경우도 resampling 처리
            legal_moves = self.get_legal_moves_movement()
            if legal_moves:
                action = random.choice(legal_moves)
                piece_index = action // 4
                direction_index = action % 4
            else:
                return self.get_state(), -1, True, {"error": "Invalid piece index and no legal moves"}
        
        di, dj = directions[direction_index]
        i, j = positions[piece_index]
        self.board[0, i, j] = 0
        
        new_i, new_j = i, j
        while True:
            next_i = new_i + di
            next_j = new_j + dj
            if not (0 <= next_i < board_size and 0 <= next_j < board_size):
                break
            if self.board[0, next_i, next_j] != 0 or self.board[1, next_i, next_j] != 0:
                break
            new_i, new_j = next_i, next_j
        
        if (new_i, new_j) == (i, j):
            # 움직임이 없는 경우에도 resampling 시도
            legal_moves = self.get_legal_moves_movement()
            if legal_moves:
                action = random.choice(legal_moves)
                return self._movement_step(action)
            else:
                return self.get_state(), -1, False, {"winner": 0}
        
        reward = 0
        done = False
        info = {}
        
        if new_i == 3 and new_j == 3:
            self.scores[self.current_player] += 1
            # print(f'score +1:{self.current_player}')
            reward = 1
        else:
            self.board[0, new_i, new_j] = 1
        
        if 3 in [new_i, new_j] and 3 not in [i, j]:
            reward = 0.5
        elif 3 not in [new_i, new_j] and 3 in [i, j]:
            reward = -0.5
        
        self.turn += 1
        
        if self.scores[self.current_player] >= 2:
            done = True
            reward = 10
            info["winner"] = self.current_player
        elif self.turn >= self.max_turns:
            done = True
            reward = -1
            info["winner"] = 0
            info["max_turn_penalty"] = True
        
        self._swap_board()
        return self.get_state(), reward + penalty, done, info
    
    def _swap_board(self):
        """보드 채널 swap 및 플레이어 교체 (항상 현재 플레이어의 말이 채널 0에 있도록)"""
        self.board = np.array([self.board[1].copy(), self.board[0].copy()])
        self.current_player = -self.current_player
        
    def render(self):
        print("Phase:", self.phase)
        print("Turn:", self.turn)
        print("Current player:", self.current_player)
        print("Scores:", self.scores)
        print("Placements:", self.placements)
        print("Board (current player's pieces, channel 0):")
        print(self.board[0])
        print("Board (opponent's pieces, channel 1):")
        print(self.board[1])
    
    # ---------------------------------------------------
    # 아래의 두 정적 메서드는 CNN/MCTS 코드와 연동하기 위해 사용됨.
    # ---------------------------------------------------
    @staticmethod
    def apply_action(state, action):
        """
        주어진 state (dictionary)에서 action을 적용하여 새로운 상태를 반환합니다.
        MCTS에서 상태 전이 함수를 호출할 때 사용합니다.
        """
        env = ExitStrategyEnv()
        # 전달받은 state로 환경 내부 상태 덮어쓰기
        env.board = copy.deepcopy(state["board"])
        env.phase = state["phase"]
        env.turn = state["turn"]
        env.current_player = state["current_player"]
        env.placements = copy.deepcopy(state["placements"])
        env.scores = copy.deepcopy(state["scores"])
        env.column_count = copy.deepcopy(state["column_count"])
        # step 함수를 통해 action 적용 (보상, 종료 여부 등은 무시하고 새 state만 반환)
        new_state, reward, _, _ = env.step(action)
        return new_state, reward

    @staticmethod
    def is_terminal(state):
        """
        주어진 state가 종료 상태인지 판단합니다.
        반환: (terminal: bool, outcome: float)
          - outcome은 현재 플레이어 관점에서 승리: 1.0, 패배: -1.0, 무승부: 0.0
        """
        board_size = state["board"].shape[1]
        phase = state["phase"]
        turn = state["turn"]
        scores = state["scores"]
        
        # 승리 조건: 플레이어 1의 점수가 2 이상이면 현재 플레이어 관점에서 승리 (채널 0가 항상 현재 플레이어)
        if scores[1] >= 2:
            return True, 1.0
        if scores[-1] >= 2:
            return True, -1.0
        if turn >= MAX_TURNS:
            return True, 0.0
        
        # movement_phase일 경우, 현재 플레이어의 말 중 하나라도 이동 가능한 칸이 있으면 계속 진행
        if phase == "movement":
            current_board = state["board"][0]
            opponent_board = state["board"][1]
            positions = np.argwhere(current_board == 1)
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            legal_exists = False
            for i, j in positions:
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < board_size and 0 <= nj < board_size:
                        if current_board[ni, nj] == 0 and opponent_board[ni, nj] == 0:
                            legal_exists = True
                            break
                if legal_exists:
                    break
            if not legal_exists:
                return True, -1.0
        return False, 0.0


#######################################
# 1. CNN 기반 Actor-Critic 네트워크 #
#######################################

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 스킵 연결
        out = F.relu(out)
        return out

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=7, in_channels=2, num_res_blocks=3, num_filters=64):
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size

        # 공유 컨볼루션 층 (트렁크)
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        # Placement phase 전용 정책 헤드 (출력 크기: 49)
        self.placement_policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.placement_policy_bn   = nn.BatchNorm2d(2)
        self.placement_policy_fc   = nn.Linear(2 * board_size * board_size, board_size * board_size)  # 7x7 = 49

        # Movement phase 전용 정책 헤드 (출력 크기: 24)
        self.movement_policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.movement_policy_bn   = nn.BatchNorm2d(2)
        self.movement_policy_fc   = nn.Linear(2 * board_size * board_size, 24)

        # 가치(value) 헤드
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(1 * board_size * board_size, 32)
        self.value_fc2  = nn.Linear(32, 1)

    def forward(self, x, legal_moves_mask=None, phase="placement"):
        # 공유 컨볼루션 트렁크
        x = F.relu(self.bn(self.conv(x)))
        for block in self.res_blocks:
            x = block(x)

        # phase에 따라 다른 정책 헤드 사용
        if phase == "placement":
            policy = F.relu(self.placement_policy_bn(self.placement_policy_conv(x)))
            policy = policy.view(policy.size(0), -1)
            policy = self.placement_policy_fc(policy)
        elif phase == "movement":
            policy = F.relu(self.movement_policy_bn(self.movement_policy_conv(x)))
            policy = policy.view(policy.size(0), -1)
            policy = self.movement_policy_fc(policy)
        else:
            raise ValueError("Unknown phase: " + phase)

        if legal_moves_mask is not None:
            # legal_moves_mask의 shape는 (batch_size, 행동 개수)여야 합니다.
            policy = policy.masked_fill(~legal_moves_mask, -1e4)
        policy = F.log_softmax(policy, dim=1)

        # 가치 헤드
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value


##########################################
# 2. MCTS 구현 (노드, 선택/확장/평가/역전파)
##########################################

class Node:
    def __init__(self, state, prior, immediate_reward = 0.0):
        self.state = state              # 현재 게임 상태 (예: 환경의 상태 dictionary)
        self.prior = prior              # 신경망이 예측한 사전 확률 P(s, a)
        self.N = 0                      # 방문 횟수
        self.W = 0.0                    # 누적 가치
        self.Q = 0.0                    # 평균 가치 (W / N)
        self.children = {}              # 자식 노드: action -> Node
        self.immediate_reward = immediate_reward

    def expand(self, action_priors, next_state_func):
        """
        action_priors: dict {action: prior probability}
        next_state_func: (state, action) -> 다음 상태를 반환하는 함수
        """
        for action, prior in action_priors.items():
            if action not in self.children:
                new_state, immediate_reward = next_state_func(self.state, action)
                self.children[action] = Node(new_state, prior, immediate_reward/10) # /10

    def update(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

def mcts_search(root, neural_net, num_simulations, c_puct, next_state_func, is_terminal_func, gamma = 1.0):
    """
    root: 초기 Node
    neural_net: 함수, state를 입력받아 (policy dict, value) 반환
    num_simulations: MCTS 시뮬레이션 횟수
    c_puct: 탐색 상수
    next_state_func: 상태 전이 함수
    is_terminal_func: (state) -> (bool, outcome)
    """
    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # 1. Selection: 트리 내려가기
        while node.children:
            best_score = -float('inf')
            best_action = None
            for action, child in node.children.items():
                score = child.Q + c_puct * child.prior * math.sqrt(node.N) / (1 + child.N)
                if score > best_score:
                    best_score = score
                    best_action = action
            node = node.children[best_action]
            search_path.append(node)
            terminal, _ = is_terminal_func(node.state)
            if terminal:
                break

        # 2. Expansion & Evaluation
        terminal, outcome = is_terminal_func(node.state)
        if terminal:
            value = outcome
        else:
            policy, value = neural_net(node.state)
            node.expand(policy, next_state_func)

        # 3. Backup: 평가 결과를 경로에 전파 (플레이어 전환이 있다면 부호 반전 필요)
        for node in reversed(search_path):
            backup_value = node.immediate_reward + gamma * value
            node.update(backup_value)
            value = -backup_value

    # 개선된 정책 분포 (방문 횟수 기반)
    total_N = sum(child.N for child in root.children.values())
    action_probs = {action: child.N / total_N for action, child in root.children.items()}
    return action_probs

##########################################
# 3. 환경 연동 관련 함수 (상태 전이, 종료 체크)
##########################################
# 상태 복제를 위해 deepcopy 사용
def next_state_func(state, action):
    new_state = copy.deepcopy(state)
    # 아래 함수는 ExitstrategyEnv에 맞게 구현되어 있어야 함.
    # 예: new_state = ExitstrategyEnv.apply_action(new_state, action)
    new_state, reward = ExitStrategyEnv.apply_action(new_state, action)
    return new_state, reward

def is_terminal_func(state):
    # ExitstrategyEnv의 종료 조건에 맞게 구현되어 있어야 함.
    # 예: return ExitstrategyEnv.is_terminal(state) → (bool, outcome)
    return ExitStrategyEnv.is_terminal(state)

##########################################
# 4. MCTS용 신경망 래퍼 함수
##########################################
def neural_net_fn(state, network, device, legal_moves_mask=None):
    board = state['board']  # (2,7,7) numpy array
    input_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(device)
    if legal_moves_mask is not None:
        legal_moves_mask = torch.tensor(legal_moves_mask, dtype=torch.bool).unsqueeze(0).to(device)
    with torch.no_grad():
        log_policy, value = network(input_tensor, legal_moves_mask=legal_moves_mask, phase=state['phase'])
    policy = torch.exp(log_policy).cpu().numpy().squeeze()
    policy_dict = {a: float(policy[a]) for a in range(policy.shape[0])}
    return policy_dict, float(value.cpu().numpy())


##########################################
# 5. Self-Play 함수: 한 에피소드 진행 및 학습 데이터 수집
##########################################
def get_legal_moves_mask(state, env):
    """
    state와 env 인스턴스를 기반으로 legal_moves_mask를 계산합니다.
    mask는 24차원 boolean 배열로, 각 index가 합법적이면 True, 아니면 False입니다.
    """
    mask = None
    if state["phase"] == "placement":
        mask = np.zeros(49, dtype=bool)
        legal_moves = env.get_legal_moves_placement()
        mask[legal_moves] = True
    elif state["phase"] == "movement":
        mask = np.zeros(24, dtype=bool)
        legal_moves = env.get_legal_moves_movement()
        mask[legal_moves] = True
    return mask

def self_play_episode(network, device, num_mcts_simulations=50, gamma=1.0):
    env = ExitStrategyEnv()
    state = env.reset()
    done = False
    episode_data = []  # 각 턴의 (state, MCTS 정책, 현재 플레이어, 즉시 reward) 저장
    
    while not done:
        legal_moves_mask = get_legal_moves_mask(state, env)
        # 현재 상태에 대해 신경망 예측 (초기 사전 확률)
        initial_policy, _ = neural_net_fn(state, network, device, legal_moves_mask)
        root = Node(state, prior=1.0)
        root.expand(initial_policy, next_state_func)
        # MCTS로 개선된 정책 분포 계산
        action_probs = mcts_search(
            root,
            lambda s: neural_net_fn(s, network, device, get_legal_moves_mask(s, env)),
            num_simulations=num_mcts_simulations,
            c_puct=1.0,
            next_state_func=next_state_func,
            is_terminal_func=is_terminal_func
        )
        
        current_player = state['current_player']
        # 행동 선택 (확률적 샘플링)
        actions = list(action_probs.keys())
        probs = np.array([action_probs[a] for a in actions])
        probs = probs / np.sum(probs)
        action = np.random.choice(actions, p=probs)
        
        # 현재 step의 즉시 reward를 기록하고, 다음 상태로 전이
        next_state, reward, done, info = env.step(action)
        episode_data.append((copy.deepcopy(state), action_probs, current_player, reward, info))
        state = next_state  # 다음 턴을 위한 상태 업데이트
    
    # 에피소드 종료 후, 각 상태에 대해 누적 reward (return)을 계산
    training_examples = []
    cumulative_return = 0.0
    penalty = -1.0
    # 뒤에서부터 누적 보상을 계산 (감마 감쇠 적용, 여기서는 gamma=1.0이면 단순 합산)
    for (s, mcts_policy, player, r, info) in reversed(episode_data):
        if info.get("max_turn_penalty", False):
            cumulative_return = r + penalty + gamma * cumulative_return
        else:
            cumulative_return = r + gamma * cumulative_return
        # 누적 reward를 해당 state에 할당합니다.
        training_examples.insert(0, (s, mcts_policy, cumulative_return))
    
    return training_examples


##########################################
# 6. 손실 함수 및 학습 루프
##########################################
def compute_loss(predicted_policy, predicted_value, target_policy, outcome, model, l2_coef=1e-4):
    # 가치 손실 (MSE)
    value_loss = F.mse_loss(predicted_value, outcome)
    # 정책 손실: log 확률과 타겟 정책의 cross entropy
    target_policy_tensor = torch.tensor(target_policy, dtype=torch.float32).to(predicted_policy.device)
    policy_loss = -torch.sum(target_policy_tensor * predicted_policy, dim=1).mean()
    # L2 정규화
    l2_loss = 0.0
    for param in model.parameters():
        l2_loss += torch.sum(param ** 2)
    return value_loss + policy_loss + l2_coef * l2_loss

def train(network, device, num_episodes=1000, num_mcts_simulations=50, initial_batch_size=32, lr=1e-3):
    optimizer = optim.Adam(network.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    replay_buffer = []
    
    # 배치 크기를 GPU 메모리에 맞게 동적으로 조정
    try:
        batch_size = initial_batch_size * 2
        torch.cuda.empty_cache()
    except RuntimeError:
        batch_size = initial_batch_size
        torch.cuda.empty_cache()
    
    for episode in range(num_episodes):
        print(f"Episode {episode+1}")
        episode_examples = self_play_episode(network, device, num_mcts_simulations)
        
        # 리플레이 버퍼 관리 개선
        if len(replay_buffer) > 10000:  # 최대 버퍼 크기 제한
            # 가장 오래된 20%의 데이터를 제거
            replay_buffer = replay_buffer[len(replay_buffer)//5:]
        replay_buffer.extend(episode_examples)
        
        # 충분한 데이터가 모이면 여러 번의 배치 업데이트 수행
        if len(replay_buffer) >= batch_size:
            total_loss = 0.0
            num_batches = min(5, len(replay_buffer) // batch_size)  # 최대 5번의 배치 업데이트
            
            for _ in range(num_batches):
                batch = random.sample(replay_buffer, batch_size)
                
                # phase별로 예제를 분리
                placement_examples = [ex for ex in batch if ex[0]["phase"] == "placement"]
                movement_examples = [ex for ex in batch if ex[0]["phase"] == "movement"]
                
                batch_loss = 0.0
                
                # Placement phase 업데이트 (병렬 처리 개선)
                if placement_examples:
                    states, target_policies, outcomes = zip(*placement_examples)
                    state_tensors = torch.stack([torch.tensor(s['board'], dtype=torch.float32) for s in states]).to(device)
                    
                    # 병렬 처리를 위한 텐서 준비
                    policy_matrices = torch.zeros((len(placement_examples), 49), dtype=torch.float32).to(device)
                    for i, policy in enumerate(target_policies):
                        for action, prob in policy.items():
                            policy_matrices[i, action] = prob
                    
                    outcome_tensors = torch.tensor(outcomes, dtype=torch.float32).view(-1, 1).to(device)
                    
                    log_policy, predicted_value = network(state_tensors, phase="placement")
                    loss = compute_loss(log_policy, predicted_value, policy_matrices, outcome_tensors, network)
                    loss.backward()
                    batch_loss += loss.item()
                
                # Movement phase 업데이트 (동일한 최적화 적용)
                if movement_examples:
                    states, target_policies, outcomes = zip(*movement_examples)
                    state_tensors = torch.stack([torch.tensor(s['board'], dtype=torch.float32) for s in states]).to(device)
                    
                    policy_matrices = torch.zeros((len(movement_examples), 24), dtype=torch.float32).to(device)
                    for i, policy in enumerate(target_policies):
                        for action, prob in policy.items():
                            policy_matrices[i, action] = prob
                    
                    outcome_tensors = torch.tensor(outcomes, dtype=torch.float32).view(-1, 1).to(device)
                    
                    log_policy, predicted_value = network(state_tensors, phase="movement")
                    loss = compute_loss(log_policy, predicted_value, policy_matrices, outcome_tensors, network)
                    loss.backward()
                    batch_loss += loss.item()
                
                optimizer.step()
                optimizer.zero_grad()
                total_loss += batch_loss
            
            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)
            print(f"Episode {episode+1}, Average Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 주기적인 모델 저장 및 평가
        if episode % 10 == 9:
            save_model(network, f"alphazero_model_ep{episode+1}.pth")
            if episode % 50 == 49:
                # 이전 버전과의 대전을 통한 성능 평가 로직 추가 가능
                pass

            
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    print(f"Model loaded from {path}")
    
##########################################
# 7. 메인 실행 부분
##########################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
  
    network = AlphaZeroNet(board_size=7, in_channels=2, num_res_blocks=3, num_filters=64).to(device)
    if os.path.exists("alphazero_model.pth"):
        load_model(network, "alphazero_model.pth", device)
    # 모델 학습
    train(network, device, num_episodes=3000, num_mcts_simulations=10, batch_size=32, lr=1e-3)
    
    # 학습 후 모델 저장
    save_model(network, "alphazero_model.pth")
    
    # 이후에 모델을 불러오려면 아래와 같이 사용 (예시)
    