import tkinter as tk
from gui import GameApp
import train_module

PENALTY_SCALE = 1

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
        self.last_actions = {1: None, 2: None}
        self.train_step()

    def draw_board(self):
        """환경의 현재 보드 상태를 tkinter Canvas에 그립니다."""
        self.canvas.delete("all")
        board = self.env.board
        cell_size = 50
        for x in range(self.env.board_size):
            for y in range(self.env.board_size):
                if x == 3 or y == 3:
                    color = "gray"
                else:
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
        한 스텝 진행:
        - 현재 에이전트가 행동을 선택하고, 환경에 적용하여 (state, action, reward, next_state)를 획득한 뒤 Q‑러닝 업데이트를 수행합니다.
        - 만약 이번 턴에서 득점(양의 보상)이 발생하면, 바로 이전에 행동한 상대 에이전트의 마지막 행동에 대해 동일한 크기의 음의 보상(페널티)을 적용합니다.
        - 에피소드가 끝났거나 winner가 정해진 경우에도 마지막 Q 업데이트 후 환경을 리셋합니다.
        """
        current_player = self.env.current_player
        agent = self.agent1 if current_player == 1 else self.agent2
        state = self.env.get_state()
        action = agent.choose_action(self.env)
        if action is None:
            # 행동 선택이 실패한 경우 환경을 리셋
            self.env.reset()
            self.current_state = self.env.get_state()
        else:
            reward, done = self.env.step(action)
            next_state = self.env.get_state()
            # 현재 에이전트에 대해 Q 업데이트

            opponent = 3 - current_player
            if self.last_actions[opponent] is not None:
                opp_state, opp_action, opp_reward = self.last_actions[opponent]
                opponent_agent = self.agent1 if opponent == 1 else self.agent2
                # 득점 발생 시 음수 보상, 아니면 0 (혹은 원하는 기본값)으로 업데이트
                penalty = -PENALTY_SCALE * reward if reward >= 100 else 0
                opponent_agent.update_q(opp_state, opp_action, opp_reward + penalty, next_state)
                # 상대방의 마지막 행동은 매 스텝마다 한 번만 업데이트하도록 초기화
                self.last_actions[opponent] = None
            

            # 현재 에이전트의 (state, action)을 마지막 행동으로 저장
            self.last_actions[current_player] = (state, action, reward)
            self.current_state = next_state

        # 만약 에피소드가 종료되었거나 winner가 결정되었다면 최종 Q 업데이트 후 환경 리셋
        if done or self.env.winner is not None:
            self.current_episode += 1
            
            # Q-table 저장
            train_module.save_q_table(self.agent1, "agent1_q.pkl")
            train_module.save_q_table(self.agent2, "agent2_q.pkl")
            # Q-table 로드하여 업데이트 (각 에이전트에 적용)
            self.agent1.q_table = train_module.load_q_table("agent1_q.pkl")
            self.agent2.q_table = train_module.load_q_table("agent2_q.pkl")

            self.env.reset()
            self.current_state = self.env.get_state()
            self.last_actions = {1: None, 2: None}

        self.draw_board()
        self.info_label.config(text=f"Episode: {self.current_episode}  Current Player: {self.env.current_player}")
        self.root.after(self.training_step_delay, self.train_step)



def start_game(player1_ai, player2_ai):
    game_window = tk.Toplevel()
    game_window.title("Exit Strategy 2")
    # 만약 두 플레이어 모두 AI이면 TrainingApp을 실행하여 학습 환경과 화면을 연결합니다.
    if player1_ai and player2_ai:
        TrainingApp(game_window, episodes=10, training_step_delay=500)
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
