import game
import tkinter as tk


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Exit Strategy 2 - AI vs Human")
    app = game.GameApp(root, ai_enabled=True)  # AI 활성화
    root.mainloop()
