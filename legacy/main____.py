import tkinter as tk
import torch
import numpy as np
import random
import copy
import os
import AlphaExitNet  # AlphaExitNet.py 파일의 모듈

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

class AlphaTrainingApp:
    def __init__(self, root, batch_size=32, training_step_delay=1000):
        self.root = root
        self.training_step_delay = training_step_delay  # 밀리초 단위 지연
        self.batch_size = batch_size

        # 환경과 네트워크 초기화
        self.env = AlphaExitNet.ExitStrategyEnv()
        self.network = AlphaExitNet.AlphaZeroNet(board_size=7, in_channels=2, num_res_blocks=3, num_filters=64)
        self.network.to(device)
        if os.path.exists("alphazero_model.pth"):
            AlphaExitNet.load_model(self.network, "alphazero_model.pth", device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)

        # 학습 데이터 저장용 변수
        self.replay_buffer = []
        self.episode_data = []  # 현재 에피소드에서 수집한 (state, mcts_policy, player, reward) 튜플
        self.episode_count = 0

        # GUI 구성요소 (캔버스, 레이블 등)
        self.canvas = tk.Canvas(root, width=350, height=350, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.info_label = tk.Label(root, text="Episode: 0", font=("Arial", 14))
        self.info_label.grid(row=1, column=0)
        self.loss_label = tk.Label(root, text="Loss: N/A", font=("Arial", 12))
        self.loss_label.grid(row=2, column=0)

        # 에피소드 시작
        self.current_state = self.env.reset()
        self.train_episode()

    def draw_board(self):
        """현재 환경의 보드 상태를 캔버스에 그립니다."""
        self.canvas.delete("all")
        board = self.env.board  # shape: (2, board_size, board_size)
        board_size = self.env.board_size
        cell_size = 50
        for x in range(board_size):
            for y in range(board_size):
                # 기본 색상: 중앙 행/열는 회색, 중앙(3,3)은 녹색
                color = "gray" if (x == 3 or y == 3) else "white"
                if (x, y) == (3, 3):
                    color = "green"
                # 채널 0에 있는 말: 현재 플레이어의 말
                if board[0, x, y] == 2:
                    color = "black"
                elif board[0, x, y] == 1:
                    color = 'blue' if self.env.current_player == 1 else 'red'
                # 채널 1에 있는 말: 상대방의 말
                elif board[1, x, y] == 1:
                    color = 'red' if self.env.current_player == 1 else 'blue'
                elif (x * board_size + y) not in self.env.get_legal_moves_placement() and self.env.phase == 'placement':
                    color = "gray"
                self.canvas.create_rectangle(
                    y * cell_size, x * cell_size,
                    y * cell_size + cell_size, x * cell_size + cell_size,
                    fill=color, outline="black"
                )

    def train_episode(self):
        """한 에피소드를 self-play로 진행합니다."""
        self.episode_data = []
        self.current_state = self.env.reset()
        self.run_step()
        

    def run_step(self):
        """에피소드 내 한 스텝을 진행한 후 GUI를 업데이트합니다."""
        state = self.current_state

        # 현재 phase에 따른 legal moves mask 계산
        legal_moves_mask = AlphaExitNet.get_legal_moves_mask(state, self.env)
        # 신경망 예측 및 초기 정책 (log policy → 확률 변환)
        initial_policy, _ = AlphaExitNet.neural_net_fn(state, self.network, device, legal_moves_mask)
        
        root_node = AlphaExitNet.Node(state, prior=1.0)
        root_node.expand(initial_policy, AlphaExitNet.next_state_func)
        # MCTS search (속도 향상을 위해 시뮬레이션 횟수 낮춤)
        action_probs = AlphaExitNet.mcts_search(
            root_node,
            lambda s: AlphaExitNet.neural_net_fn(s, self.network, device, AlphaExitNet.get_legal_moves_mask(s, self.env)),
            num_simulations=15,
            c_puct=1.0,
            next_state_func=AlphaExitNet.next_state_func,
            is_terminal_func=AlphaExitNet.is_terminal_func
        )
        # 확률에 따라 행동 선택
        # action_probs = initial_policy

        actions = list(action_probs.keys())
        probs = np.array([action_probs[a] for a in actions])
        probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones_like(probs) / len(probs)
        action = np.random.choice(actions, p=probs)

        # 행동 적용 및 결과 업데이트
        next_state, reward, done, info = self.env.step(action)
        self.episode_data.append((copy.deepcopy(state), action_probs, state["current_player"], reward, info))
        self.current_state = next_state

        # 보드 및 정보 업데이트
        self.draw_board()
        self.info_label.config(text=f"Episode: {self.episode_count}  Phase: {state['phase']}")

        if done:
            # 에피소드 종료 시 학습 데이터 처리 및 네트워크 업데이트
            self.episode_count += 1
            self.process_episode()
        else:
            # 다음 스텝 진행 (지연 후 호출)
            self.root.after(self.training_step_delay, self.run_step)

    def process_episode(self):
        """한 에피소드 종료 후 누적 리턴 계산 및 배치 업데이트를 진행합니다."""
        cumulative_return = 0.0
        episode_examples = []
        gamma = 1.0  # 감쇠 계수 (여기서는 단순 합산)
        penalty = -1.0
        for (s, mcts_policy, player, r, info) in reversed(self.episode_data):
            if info.get("max_turn_penalty", False):
                cumulative_return = r + penalty + gamma * cumulative_return
            else:
                cumulative_return = r + gamma * cumulative_return
            episode_examples.insert(0, (s, mcts_policy, cumulative_return))
        self.replay_buffer.extend(episode_examples)

        loss_val = None
        # 배치 사이즈 이상이면 업데이트 진행
        if len(self.replay_buffer) >= self.batch_size:
            batch_size = int(len(self.replay_buffer) * 0.5)
            batch = random.sample(self.replay_buffer, batch_size)
            total_loss = 0.0

            # placement phase 예제 업데이트 (행동 차원: 49)
            placement_examples = [ex for ex in batch if ex[0]["phase"] == "placement"]
            if placement_examples:
                states, target_policies, outcomes = zip(*placement_examples)
                state_tensors = torch.stack([torch.tensor(s['board'], dtype=torch.float32) for s in states]).to(device)
                target_policy_tensors = torch.stack([
                    torch.tensor([target_policies[i].get(a, 0.0) for a in range(49)], dtype=torch.float32)
                    for i in range(len(placement_examples))
                ]).to(device)
                outcome_tensors = torch.tensor(outcomes, dtype=torch.float32).view(-1, 1).to(device)
                log_policy, predicted_value = self.network(state_tensors, phase="placement")
                loss = AlphaExitNet.compute_loss(log_policy, predicted_value, target_policy_tensors, outcome_tensors, self.network)
                loss.backward()
                total_loss += loss.item()

            # movement phase 예제 업데이트 (행동 차원: 24)
            movement_examples = [ex for ex in batch if ex[0]["phase"] == "movement"]
            if movement_examples:
                states, target_policies, outcomes = zip(*movement_examples)
                state_tensors = torch.stack([torch.tensor(s['board'], dtype=torch.float32) for s in states]).to(device)
                target_policy_tensors = torch.stack([
                    torch.tensor([target_policies[i].get(a, 0.0) for a in range(24)], dtype=torch.float32)
                    for i in range(len(movement_examples))
                ]).to(device)
                outcome_tensors = torch.tensor(outcomes, dtype=torch.float32).view(-1, 1).to(device)
                log_policy, predicted_value = self.network(state_tensors, phase="movement")
                loss = AlphaExitNet.compute_loss(log_policy, predicted_value, target_policy_tensors, outcome_tensors, self.network)
                loss.backward()
                total_loss += loss.item()

            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_val = total_loss
            self.replay_buffer = []  # 버퍼 초기화

        # if self.episode_count % 5 == 4:
        AlphaExitNet.save_model(self.network, "alphazero_model.pth")

        if loss_val is not None:
            self.loss_label.config(text=f"Loss: {loss_val:.4f}")

        # 다음 에피소드를 일정 지연 후 시작
        self.root.after(self.training_step_delay, self.train_episode)

import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from concurrent.futures import ThreadPoolExecutor
import queue

class OptimizedAlphaTrainingApp(AlphaTrainingApp):
    def __init__(self, root, batch_size=32, training_step_delay=1000):
        super().__init__(root, batch_size, training_step_delay)
        self.scaler = torch.GradScaler("cuda") if torch.cuda.is_available() else torch.GradScaler("mps")  # Mixed precision training
        self.episode_queue = queue.Queue()
        self.training_executor = ThreadPoolExecutor(max_workers=1)
        self.mcts_executor = ThreadPoolExecutor(max_workers=4)
        
    def run_parallel_mcts(self, state):
        """병렬 MCTS 실행"""
        futures = []
        for _ in range(4):  # 4개의 병렬 MCTS 실행
            futures.append(self.mcts_executor.submit(
                AlphaExitNet.mcts_search,
                root_node=AlphaExitNet.Node(state, prior=1.0),
                network_fn=lambda s: AlphaExitNet.neural_net_fn(s, self.network, device, 
                    AlphaExitNet.get_legal_moves_mask(s, self.env)),
                num_simulations=4,  # 각 MCTS는 더 적은 시뮬레이션 수행
                c_puct=1.0
            ))
        
        # 결과 집계
        action_probs = {}
        for future in futures:
            probs = future.result()
            for action, prob in probs.items():
                action_probs[action] = action_probs.get(action, 0) + prob / len(futures)
        
        return action_probs

    def optimize_network(self, batch):
        """최적화된 네트워크 학습"""
        with torch.autocast('cuda'):  # Mixed precision training
            # Placement phase
            placement_examples = [ex for ex in batch if ex[0]["phase"] == "placement"]
            if placement_examples:
                states, policies, outcomes = zip(*placement_examples)
                state_tensors = torch.stack([
                    torch.tensor(s['board'], dtype=torch.float32, pin_memory=True) 
                    for s in states
                ]).to(device, non_blocking=True)
                
                policy_tensors = torch.stack([
                    torch.tensor([policies[i].get(a, 0.0) for a in range(49)], 
                               dtype=torch.float32, pin_memory=True)
                    for i in range(len(placement_examples))
                ]).to(device, non_blocking=True)
                
                outcome_tensors = torch.tensor(
                    outcomes, dtype=torch.float32, pin_memory=True
                ).view(-1, 1).to(device, non_blocking=True)
                
                log_policy, predicted_value = self.network(state_tensors, phase="placement")
                loss = AlphaExitNet.compute_loss(
                    log_policy, predicted_value, policy_tensors, outcome_tensors, self.network
                )
                self.scaler.scale(loss).backward()

            # Movement phase (similar optimization)
            movement_examples = [ex for ex in batch if ex[0]["phase"] == "movement"]
            if movement_examples:
                states, policies, outcomes = zip(*movement_examples)
                state_tensors = torch.stack([
                    torch.tensor(s['board'], dtype=torch.float32, pin_memory=True) 
                    for s in states
                ]).to(device, non_blocking=True)
                
                policy_tensors = torch.stack([
                    torch.tensor([policies[i].get(a, 0.0) for a in range(24)], 
                               dtype=torch.float32, pin_memory=True)
                    for i in range(len(movement_examples))
                ]).to(device, non_blocking=True)
                
                outcome_tensors = torch.tensor(
                    outcomes, dtype=torch.float32, pin_memory=True
                ).view(-1, 1).to(device, non_blocking=True)
                
                log_policy, predicted_value = self.network(state_tensors, phase="movement")
                loss = AlphaExitNet.compute_loss(
                    log_policy, predicted_value, policy_tensors, outcome_tensors, self.network
                )
                self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def run_step(self):
        """최적화된 스텝 실행"""
        def training_loop():
            state = self.current_state
            action_probs = self.run_parallel_mcts(state)
            
            # 행동 선택 및 환경 스텝
            actions = list(action_probs.keys())
            probs = np.array([action_probs[a] for a in actions])
            probs = probs / np.sum(probs)
            action = np.random.choice(actions, p=probs)
            
            next_state, reward, done, info = self.env.step(action)
            self.episode_data.append((
                copy.deepcopy(state), action_probs, 
                state["current_player"], reward, info
            ))
            
            return next_state, done

        # 비동기적으로 학습 스텝 실행
        future = self.training_executor.submit(training_loop)
        next_state, done = future.result()
        
        self.current_state = next_state
        self.draw_board()
        self.info_label.config(text=f"Episode: {self.episode_count}  Phase: {self.env.phase}")

        if done:
            self.episode_count += 1
            self.process_episode()
        else:
            self.root.after(self.training_step_delay, self.run_step)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("AlphaExitNet Training GUI")
    app = OptimizedAlphaTrainingApp(root, batch_size=32, training_step_delay=1)
    root.mainloop()
