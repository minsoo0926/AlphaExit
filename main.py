import tkinter as tk
import torch
import numpy as np
import random
import copy
import os
import threading
from queue import Queue
import AlphaExitNet

model = 'cuda' if torch.cuda.is_available() else 'mps'

class GPUOptimizedAlphaTrainingApp:
    def __init__(self, root, batch_size=128, training_step_delay=1):
        self.root = root
        self.training_step_delay = training_step_delay
        self.batch_size = batch_size
        
        # GPU 관련 설정
        self.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # CUDNN 최적화
        
        # 환경과 네트워크 초기화
        self.env = AlphaExitNet.ExitStrategyEnv()
        self.network = AlphaExitNet.AlphaZeroNet(
            board_size=7,
            in_channels=2,
            num_res_blocks=3,  # 기존과 동일하게 유지
            num_filters=64     # 기존과 동일하게 유지
        )
        self.network.to(self.device)
        if os.path.exists("alphazero_model.pth"):
            AlphaExitNet.load_model(self.network, "alphazero_model.pth", self.device)
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=2e-4)
        
        # 학습 데이터 관련 변수
        self.replay_buffer = []
        self.episode_data = []
        self.episode_count = 0
        self.max_replay_buffer_size = 10000  # 메모리 관리를 위한 최대 버퍼 크기
        
        # MCTS 관련 설정
        self.num_simulations = 100  # GPU에서는 더 많은 시뮬레이션 가능
        self.temperature = 1.0  # 초기 탐험 온도
        
        # GUI 구성요소
        self.canvas = tk.Canvas(root, width=350, height=350, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.info_label = tk.Label(root, text="Episode: 0", font=("Arial", 14))
        self.info_label.grid(row=1, column=0)
        self.loss_label = tk.Label(root, text="Loss: N/A", font=("Arial", 12))
        self.loss_label.grid(row=2, column=0)
        self.phase_label = tk.Label(root, text="Phase: placement", font=("Arial", 12))
        self.phase_label.grid(row=3, column=0)
        
        # 상태 업데이트를 위한 큐
        self.state_queue = Queue()
        
        # 초기 상태 설정
        self.current_state = self.env.reset()
        self.start_training_thread()
        self.update_gui()

    def draw_board(self):
        """기존 보드 시각화 로직 유지"""
        self.canvas.delete("all")
        board = self.env.board
        board_size = self.env.board_size
        cell_size = 50
        
        for x in range(board_size):
            for y in range(board_size):
                color = "gray" if (x == 3 or y == 3) else "white"
                if (x, y) == (3, 3):
                    color = "green"
                if board[0, x, y] == 2:
                    color = "black"
                elif board[0, x, y] == 1:
                    color = 'blue' if self.env.current_player == 1 else 'red'
                elif board[1, x, y] == 1:
                    color = 'red' if self.env.current_player == 1 else 'blue'
                elif (x * board_size + y) not in self.env.get_legal_moves_placement() and self.env.phase == 'placement':
                    color = "gray"
                    
                self.canvas.create_rectangle(
                    y * cell_size, x * cell_size,
                    y * cell_size + cell_size, x * cell_size + cell_size,
                    fill=color, outline="black"
                )

    def update_gui(self):
        """GUI 업데이트 처리"""
        try:
            while not self.state_queue.empty():
                state, episode_count, loss, phase = self.state_queue.get_nowait()
                self.current_state = state
                self.draw_board()
                self.info_label.config(text=f"Episode: {episode_count}")
                if loss is not None:
                    self.loss_label.config(text=f"Loss: {loss:.4f}")
                self.phase_label.config(text=f"Phase: {phase}")
        except:
            pass
        finally:
            self.root.after(self.training_step_delay, self.update_gui)

    def run_mcts(self, state):
        """GPU에 최적화된 MCTS 실행"""
        legal_moves_mask = AlphaExitNet.get_legal_moves_mask(state, self.env)
        with torch.no_grad():
            initial_policy, _ = AlphaExitNet.neural_net_fn(state, self.network, self.device, legal_moves_mask)
        
        root_node = AlphaExitNet.Node(state, prior=1.0)
        root_node.expand(initial_policy, AlphaExitNet.next_state_func)
        
        action_probs = AlphaExitNet.mcts_search(
            root_node,
            lambda s: AlphaExitNet.neural_net_fn(s, self.network, self.device, AlphaExitNet.get_legal_moves_mask(s, self.env)),
            num_simulations=self.num_simulations,
            c_puct=1.0,
            next_state_func=AlphaExitNet.next_state_func,
            is_terminal_func=AlphaExitNet.is_terminal_func
        )
        return action_probs

    def train_network(self, batch):
        """GPU에 최적화된 네트워크 학습"""
        total_loss = 0
        self.optimizer.zero_grad()
        
        # Placement phase
        placement_examples = [ex for ex in batch if ex[0]["phase"] == "placement"]
        if placement_examples:
            states, policies, outcomes = zip(*placement_examples)
            state_tensors = torch.stack([
                torch.tensor(s['board'], dtype=torch.float32) 
                for s in states
            ]).to(self.device)
            policy_tensors = torch.stack([
                torch.tensor([policies[i].get(a, 0.0) for a in range(49)], dtype=torch.float32)
                for i in range(len(placement_examples))
            ]).to(self.device)
            outcome_tensors = torch.tensor(outcomes, dtype=torch.float32).view(-1, 1).to(self.device)
            
            log_policy, predicted_value = self.network(state_tensors, phase="placement")
            loss = AlphaExitNet.compute_loss(log_policy, predicted_value, policy_tensors, outcome_tensors, self.network)
            loss.backward()
            total_loss += loss.item()

        # Movement phase
        movement_examples = [ex for ex in batch if ex[0]["phase"] == "movement"]
        if movement_examples:
            states, policies, outcomes = zip(*movement_examples)
            state_tensors = torch.stack([
                torch.tensor(s['board'], dtype=torch.float32) 
                for s in states
            ]).to(self.device)
            policy_tensors = torch.stack([
                torch.tensor([policies[i].get(a, 0.0) for a in range(24)], dtype=torch.float32)
                for i in range(len(movement_examples))
            ]).to(self.device)
            outcome_tensors = torch.tensor(outcomes, dtype=torch.float32).view(-1, 1).to(self.device)
            
            log_policy, predicted_value = self.network(state_tensors, phase="movement")
            loss = AlphaExitNet.compute_loss(log_policy, predicted_value, policy_tensors, outcome_tensors, self.network)
            loss.backward()
            total_loss += loss.item()

        self.optimizer.step()
        return total_loss

    def training_loop(self):
        """별도 스레드에서 실행되는 학습 루프"""
        while True:
            state = self.env.reset()
            self.episode_data = []
            
            while True:
                action_probs = self.run_mcts(state)
                
                # 행동 선택 (temperature 적용)
                actions = list(action_probs.keys())
                probs = np.array([action_probs[a] for a in actions])
                if self.episode_count < 500:  # 초기에는 더 많은 탐험
                    probs = probs ** (1 / self.temperature)
                probs = probs / np.sum(probs)
                action = np.random.choice(actions, p=probs)
                
                next_state, reward, done, info = self.env.step(action)
                self.episode_data.append((
                    copy.deepcopy(state),
                    action_probs,
                    state["current_player"],
                    reward,
                    info
                ))
                
                # GUI 업데이트를 위한 상태 전송
                self.state_queue.put((next_state, self.episode_count, None, state["phase"]))
                
                if done:
                    break
                state = next_state
            
            # 에피소드 종료 후 처리
            self.episode_count += 1
            self.process_episode()
            
            # 모델 저장
            if self.episode_count % 10 == 0:
                AlphaExitNet.save_model(self.network, "alphazero_model.pth")
                
            # Temperature 조정
            if self.episode_count % 100 == 0:
                self.temperature = max(0.1, self.temperature * 0.95)

    def process_episode(self):
        """에피소드 데이터 처리 및 학습"""
        # 누적 보상 계산
        cumulative_return = 0.0
        gamma = 1.0
        penalty = -1.0
        
        for (s, mcts_policy, player, r, info) in reversed(self.episode_data):
            if info.get("max_turn_penalty", False):
                cumulative_return = r + penalty + gamma * cumulative_return
            else:
                cumulative_return = r + gamma * cumulative_return
            self.replay_buffer.insert(0, (s, mcts_policy, cumulative_return))
        
        # 버퍼 크기 제한
        if len(self.replay_buffer) > self.max_replay_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_replay_buffer_size:]
        
        # 배치 학습
        if len(self.replay_buffer) >= self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)
            loss = self.train_network(batch)
            self.state_queue.put((self.current_state, self.episode_count, loss, self.env.phase))

    def start_training_thread(self):
        """학습 스레드 시작"""
        training_thread = threading.Thread(target=self.training_loop, daemon=True)
        training_thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("AlphaExitNet GPU Training")
    app = GPUOptimizedAlphaTrainingApp(root)
    root.mainloop()