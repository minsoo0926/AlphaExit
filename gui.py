# gui.py
import tkinter as tk
from tkinter import messagebox
import numpy as np
from board_game import BoardGame
from rl_agent import QLearningAgent, load_q_table

class GameApp:
    def __init__(self, root, player1_ai=False, player2_ai=False):
        self.root = root
        self.player1_ai = player1_ai
        self.player2_ai = player2_ai
        # AI 플레이어이면 저장된 Q-table을 불러옵니다.
        self.agent1 = QLearningAgent(q_table=load_q_table("agent1_q.pkl")) if player1_ai else None
        self.agent2 = QLearningAgent(q_table=load_q_table("agent2_q.pkl")) if player2_ai else None

        self.game = BoardGame()
        self.canvas = tk.Canvas(root, width=350, height=350, bg="white")
        self.canvas.grid(row=1, column=0, columnspan=3)
        self.status_label = tk.Label(root, text="Player 1: 말 배치 단계", font=("Arial", 14))
        self.status_label.grid(row=2, column=0, columnspan=3)
        self.reset_button = tk.Button(root, text="Reset", command=self.reset_game)
        self.reset_button.grid(row=3, column=1)
        self.canvas.bind("<Button-1>", self.click_board)
        self.draw_board()

        # 게임 시작 시, AI 턴이면 자동 실행
        self.root.after(500, self.check_ai_turn)

    def draw_board(self):
        self.canvas.delete("all")
        for x in range(7):
            for y in range(7):
                color = "white"
                if (x, y) == self.game.green_zone:
                    color = "green"
                elif self.game.board[x, y] == -1:
                    color = "gray"
                elif self.game.board[x, y] == -2:
                    color = "black"
                elif self.game.board[x, y] == 1:
                    color = "blue"
                elif self.game.board[x, y] == 2:
                    color = "red"
                self.canvas.create_rectangle(
                    y * 50, x * 50, y * 50 + 50, x * 50 + 50,
                    fill=color, outline="black"
                )

    def click_board(self, event):
        x, y = event.y // 50, event.x // 50
        if self.game.winner:
            messagebox.showinfo("Game Over", f"Player {self.game.winner} wins!")
            return

        # 현재 턴이 AI이면 플레이어 입력 무시
        if (self.game.current_player == 1 and self.agent1 is not None) or \
           (self.game.current_player == 2 and self.agent2 is not None):
            return

        if self.game.placement_phase:
            if self.game.is_valid_placement(x, y) and self.game.board[x, y] == 0:
                if self.game.place_piece(x, y):
                    self.draw_board()
                    self.status_label.config(text=f"Player {self.game.current_player}: 말 배치 단계")
            else:
                self.status_label.config(text=f"Player {self.game.current_player}: 잘못된 위치")
        else:
            # 이동 단계: 클릭으로 말 선택 및 이동
            if self.game.board[x, y] == self.game.current_player:
                self.game.selected_piece = (x, y)
                self.status_label.config(text=f"Player {self.game.current_player}: 말 선택됨")
            elif self.game.selected_piece:
                start = self.game.selected_piece
                dx = x - start[0]
                dy = y - start[1]
                if (dx != 0 and dy != 0) or (dx == 0 and dy == 0):
                    self.status_label.config(text="잘못된 이동 방향")
                    return
                direction = (dx // abs(dx), 0) if dx != 0 else (0, dy // abs(dy))
                if self.game.move_piece(start, direction):
                    self.draw_board()
                    self.game.selected_piece = None
                    self.status_label.config(text=f"Player {self.game.current_player} 차례")
                else:
                    self.status_label.config(text="이동 불가")
            else:
                self.status_label.config(text="먼저 말을 선택하세요")

    def ai_move(self):
        # 현재 턴의 플레이어가 AI인 경우에만 AI가 행동하도록 함
        agent = None
        if self.game.current_player == 1 and self.agent1 is not None:
            agent = self.agent1
        elif self.game.current_player == 2 and self.agent2 is not None:
            agent = self.agent2
        if not agent:
            return

        state = self.game.board.copy()
        action = agent.get_best_action(state)
        if action:
            if self.game.placement_phase:
                self.game.place_piece(*action)
            else:
                x, y, nx, ny = action
                direction = (nx - x, ny - y)
                self.game.move_piece((x, y), direction)
            self.draw_board()
            self.status_label.config(text=f"Player {self.game.current_player} 차례")

    def check_ai_turn(self):
        # AI 턴이면 자동으로 ai_move를 호출
        if self.game.winner is None:
            if (self.game.current_player == 1 and self.agent1 is not None) or \
               (self.game.current_player == 2 and self.agent2 is not None):
                self.ai_move()
        self.root.after(500, self.check_ai_turn)

    def reset_game(self):
        self.game = BoardGame()
        self.draw_board()
        self.status_label.config(text="Player 1: 말 배치 단계")
