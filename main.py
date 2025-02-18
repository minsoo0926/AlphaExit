import tkinter as tk
from gui import GameApp
import train_module

class TrainingApp:
    """
    두 AI 에이전트가 self-play로 학습하는 환경을 tkinter 화면에서 실시간으로 확인할 수 있도록 구성합니다.
    """
    def __init__(self, root, episodes=1000, training_step_delay=1000):
        self.root = root
        self.episodes = episodes
        self.training_step_delay = training_step_delay  # 한 스텝당 대기 시간 (ms)
        self.env = train_module.ExitStrategyEnv()  # 학습 전용 환경
        # 기존 학습 파일이 존재하면 Q-table을 로드합니다.
        self.agent1 = train_module.QLearningAgent(
            player=1, 
            epsilon=0.1, 
            q_table=train_module.load_q_table("agent1_q.pkl")
        )
        self.agent2 = train_module.QLearningAgent(
            player=2, 
            epsilon=0.1, 
            q_table=train_module.load_q_table("agent2_q.pkl")
        )
        self.canvas = tk.Canvas(root, width=350, height=350, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.info_label = tk.Label(root, text="Episode: 0", font=("Arial", 14))
        self.info_label.grid(row=1, column=0)
        self.current_episode = 0
        self.current_state = self.env.reset()
        self.train_step()

    def draw_board(self):
        """환경의 현재 보드 상태를 tkinter Canvas에 그립니다."""
        self.canvas.delete("all")
        board = self.env.board
        cell_size = 50
        for x in range(self.env.board_size):
            for y in range(self.env.board_size):
                color = "white"
                if (x, y) == self.env.green_zone:
                    color = "green"
                elif board[x, y] == -2:
                    color = "black"
                elif board[x, y] == -1:
                    color = "gray"
                elif board[x, y] == 1:
                    color = "blue"
                elif board[x, y] == 2:
                    color = "red"
                self.canvas.create_rectangle(
                    y * cell_size, x * cell_size,
                    y * cell_size + cell_size, x * cell_size + cell_size,
                    fill=color, outline="black"
                )

    def train_step(self):
        """
        한 스텝 진행합니다.
        환경에서 현재 플레이어의 행동을 에이전트가 선택하고 실행하며, Q값 업데이트 후 보드를 갱신합니다.
        에피소드가 종료되면 환경을 리셋하고 Q-table을 파일에 저장합니다.
        """
        if self.env.winner is not None:
            # 에피소드 종료: 에피소드 수를 증가시키고 Q-table을 저장한 후 환경 리셋
            self.current_episode += 1
            
            # Q-table 업데이트: 각 에이전트의 Q-table을 파일로 저장합니다.
            train_module.save_q_table(self.agent1, "agent1_q.pkl")
            train_module.save_q_table(self.agent2, "agent2_q.pkl")
            
            if self.current_episode >= self.episodes:
                self.info_label.config(text=f"Training finished at episode {self.current_episode}")
                return
            else:
                self.env.reset()
                self.current_state = self.env.get_state()
        
        current_player = self.env.current_player
        agent = self.agent1 if current_player == 1 else self.agent2
        action = agent.choose_action(self.env)
        if action is None:
            # 유효한 행동이 없으면 에피소드 종료 처리
            self.env.reset()
            self.current_state = self.env.get_state()
        else:
            reward, done = self.env.step(action)
            next_state = self.env.get_state()
            agent.update_q(self.current_state, action, reward, next_state)
            self.current_state = next_state

        self.draw_board()
        self.info_label.config(text=f"Episode: {self.current_episode}  Current Player: {self.env.current_player}")
        self.root.after(self.training_step_delay, self.train_step)


def start_game(player1_ai, player2_ai):
    game_window = tk.Toplevel()
    game_window.title("Exit Strategy 2")
    # 만약 두 플레이어 모두 AI이면 TrainingApp을 실행하여 학습 환경과 화면을 연결합니다.
    if player1_ai and player2_ai:
        TrainingApp(game_window, episodes=10, training_step_delay=5)
    else:
        GameApp(game_window, player1_ai, player2_ai)

def on_start():
    p1_ai = (player1_var.get() == "AI")
    p2_ai = (player2_var.get() == "AI")
    start_game(p1_ai, p2_ai)

root = tk.Tk()
root.title("게임 설정")

# 플레이어 타입 선택 (라디오 버튼)
player1_var = tk.StringVar(value="Human")
player2_var = tk.StringVar(value="Human")

tk.Label(root, text="Player 1:").grid(row=0, column=0, padx=10, pady=10)
tk.Radiobutton(root, text="Human", variable=player1_var, value="Human").grid(row=0, column=1)
tk.Radiobutton(root, text="AI", variable=player1_var, value="AI").grid(row=0, column=2)

tk.Label(root, text="Player 2:").grid(row=1, column=0, padx=10, pady=10)
tk.Radiobutton(root, text="Human", variable=player2_var, value="Human").grid(row=1, column=1)
tk.Radiobutton(root, text="AI", variable=player2_var, value="AI").grid(row=1, column=2)

tk.Button(root, text="게임 시작", command=on_start).grid(row=2, column=1, pady=20)

root.mainloop()
