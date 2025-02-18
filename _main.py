# main.py
import tkinter as tk
from gui import GameApp

def start_game(player1_ai, player2_ai):
    game_window = tk.Toplevel()
    game_window.title("Exit Strategy 2")
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
