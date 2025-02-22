import tkinter as tk
from gui import GameApp
import train_module
import torch
import numpy as np

PENALTY_SCALE = 1

class TrainingApp:
    def __init__(self, root, episodes=1000, training_step_delay=1000):
        self.root = root
        self.episodes = episodes
        self.training_step_delay = training_step_delay
        self.env = train_module.ExitStrategyEnv()
        
        self.agent1 = train_module.DQNAgent(player=1)
        self.agent2 = train_module.DQNAgent(player=2)
        #  # Load trained models if available
        # try:
        #     self.agent1.model.load_state_dict(torch.load("agent1_dqn.pth", weights_only=True))
        #     self.agent1.model.eval()
        #     print("Loaded agent1_dqn.pth")
        # except FileNotFoundError:
        #     print("No saved model for agent1")
        
        # try:
        #     self.agent2.model.load_state_dict(torch.load("agent2_dqn.pth", weights_only=True))
        #     self.agent2.model.eval()
        #     print("Loaded agent2_dqn.pth")
        # except FileNotFoundError:
        #     print("No saved model for agent2")
            
        self.canvas = tk.Canvas(root, width=350, height=350, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.info_label = tk.Label(root, text="Episode: 0", font=("Arial", 14))
        self.info_label.grid(row=1, column=0)
        self.current_episode = 0
        self.current_state = self.env.reset()
        self.last_actions = {1: None, 2: None}
        self.train_step()

    def draw_board(self):
        self.canvas.delete("all")
        board = self.env.board
        cell_size = 50
        for x in range(self.env.board_size):
            for y in range(self.env.board_size):
                color = "gray" if (x == 3 or y == 3) else "white"
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
        current_player = self.env.current_player
        agent = self.agent1 if current_player == 1 else self.agent2
        state = self.env.get_state()
        action = agent.choose_action(self.env)
        opponent = self.agent2 if current_player == 1 else self.agent1
        
        if action is None:
            self.env.reset()
            self.current_state = self.env.get_state()
            done = True
        else:
            reward, done = self.env.step(action)
            next_state = self.env.get_state()
            if opponent.last_experience:
                _, _, opponent_reward = opponent.last_experience
            else:
                opponent_reward = 0.0
            agent.update(state, action, reward, opponent_reward, done)
            self.current_state = next_state

        if self.env.winner is not None:
            print(f'winner: {self.env.winner}')
            opponent.update(np.zeros(51), np.zeros(4), 0, 1, done)
            self.current_episode += 1
            self.env.reset()
            torch.save(self.agent1.target_model.state_dict(), "agent1_dqn.pth")
            torch.save(self.agent2.target_model.state_dict(), "agent2_dqn.pth")
            self.current_state = self.env.get_state()
            self.last_actions = {1: None, 2: None}
        elif done:
            self.current_episode += 1
            self.env.reset()
            torch.save(self.agent1.target_model.state_dict(), "agent1_dqn.pth")
            torch.save(self.agent2.target_model.state_dict(), "agent2_dqn.pth")
            self.current_state = self.env.get_state()
            self.last_actions = {1: None, 2: None}

        self.draw_board()
        self.info_label.config(text=f"Episode: {self.current_episode}  Current Player: {self.env.current_player}")
        self.root.after(self.training_step_delay, self.train_step)



def start_game(player1_ai, player2_ai):
    game_window = tk.Toplevel()
    game_window.title("Exit Strategy 2")
    if player1_ai and player2_ai:
        TrainingApp(game_window, episodes=100, training_step_delay=1)
    else:
        GameApp(game_window, player1_ai, player2_ai)

def on_start():
    p1_ai = (player1_var.get() == "AI")
    p2_ai = (player2_var.get() == "AI")
    start_game(p1_ai, p2_ai)

root = tk.Tk()
root.title("게임 설정")

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
