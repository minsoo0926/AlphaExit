# board_game.py
import numpy as np
from tkinter import messagebox

class Piece:
    def __init__(self, player, number, x, y):
        self.player = player
        self.number = number
        self.x = x
        self.y = y

class BoardGame:
    def __init__(self):
        self.board = np.zeros((7, 7), dtype=int)
        self.green_zone = (3, 3)
        self.forbidden_rows = [3]
        self.forbidden_cols = [3]
        self.obstacles = [(3, 0), (3, 6)]
        self.player_pieces = {1: [], 2: []}
        self.current_player = 1
        self.placement_phase = True
        self.piece_counter = {1: 0, 2: 0}
        self.column_counts = {1: [0] * 7, 2: [0] * 7}
        self.scores = {1: 0, 2: 0}
        self.winner = None
        self.selected_piece = None
        self.history = []
        self.init_board()

    def init_board(self):
        for x in range(7):
            for y in range(7):
                if not self.is_valid_placement(x, y):
                    self.board[x, y] = -1
        for ox, oy in self.obstacles:
            self.board[ox, oy] = -2

    def is_valid_placement(self, x, y):
        if (x, y) in self.obstacles or (x == 3 or y == 3):
            return False
        if self.current_player == 1 and (y < 0 or y > 2):
            return False
        if self.current_player == 2 and (y < 4 or y > 6):
            return False
        if self.column_counts[self.current_player][y] >= 2:
            return False
        return self.board[x, y] == 0

    def place_piece(self, x, y):
        if self.is_valid_placement(x, y):
            piece_number = len(self.player_pieces[self.current_player]) + 1
            piece = Piece(self.current_player, piece_number, x, y)
            self.player_pieces[self.current_player].append(piece)
            self.board[x, y] = self.current_player
            self.piece_counter[self.current_player] += 1
            self.column_counts[self.current_player][y] += 1
            self.history.append((self.board.copy(), self.current_player, self.scores.copy(),
                                 self.piece_counter.copy(), self.winner,
                                 [[Piece(p.player, p.number, p.x, p.y) for p in pieces]
                                  for pieces in self.player_pieces.values()]))
            if self.piece_counter[1] == 6 and self.piece_counter[2] == 6:
                self.placement_phase = False
                self.reset_board_after_placement()
                self.current_player = 1
            else:
                self.current_player = 3 - self.current_player
                self.reset_placement_tiles()
            return True
        return False

    def reset_board_after_placement(self):
        for x in range(7):
            for y in range(7):
                if (x, y) == self.green_zone or (x, y) in self.obstacles or (x == 3 or y == 3):
                    continue
                if self.board[x, y] not in [1, 2]:
                    self.board[x, y] = 0

    def reset_placement_tiles(self):
        for x in range(7):
            for y in range(7):
                if (x, y) in self.obstacles or (x == 3 or y == 3):
                    continue
                if any(piece.x == x and piece.y == y for pieces in self.player_pieces.values() for piece in pieces):
                    continue
                self.board[x, y] = 0 if self.board[x, y] == -1 else self.board[x, y]
                if self.is_valid_placement(x, y):
                    self.board[x, y] = 0
                else:
                    self.board[x, y] = -1

    def move_piece(self, start, direction):
        if self.winner:
            return False
        x, y = start
        player = self.board[x, y]
        if player != self.current_player:
            return False
        dx, dy = direction
        self.history.append((self.board.copy(), self.current_player, self.scores.copy(),
                             self.piece_counter.copy(), self.winner,
                             [[Piece(p.player, p.number, p.x, p.y) for p in pieces]
                              for pieces in self.player_pieces.values()]))
        while 0 <= x + dx < 7 and 0 <= y + dy < 7:
            nx, ny = x + dx, y + dy
            if self.board[nx, ny] in [-2, 1, 2]:
                break
            x, y = nx, ny
        self.board[start[0], start[1]] = -1 if start[0] == 3 or start[1] == 3 else 0
        if (x, y) == self.green_zone:
            self.scores[self.current_player] += 1
            self.check_winner()
            self.player_pieces[self.current_player] = [piece for piece in self.player_pieces[self.current_player]
                                                         if not (piece.x == start[0] and piece.y == start[1])]
        else:
            self.board[x, y] = player
            for piece in self.player_pieces[self.current_player]:
                if piece.x == start[0] and piece.y == start[1]:
                    piece.x, piece.y = x, y
        self.current_player = 3 - self.current_player
        return True

    def check_winner(self):
        if self.scores[1] == 2:
            self.winner = 1
            messagebox.showinfo("Game Over", "Player 1 wins!")
        elif self.scores[2] == 2:
            self.winner = 2
            messagebox.showinfo("Game Over", "Player 2 wins!")

    def get_possible_moves(self, position):
        x, y = position
        directions = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x, y
            while 0 <= nx + dx < 7 and 0 <= ny + dy < 7:
                nx, ny = nx + dx, ny + dy
                if self.board[nx, ny] in [-2, 1, 2]:
                    break
                directions.append(((dx, dy), (nx, ny)))
        return directions

    def undo(self):
        if self.history:
            state = self.history.pop()
            self.board, self.current_player, self.scores, self.piece_counter, self.winner, restored_pieces = state
            self.player_pieces = {1: restored_pieces[0], 2: restored_pieces[1]}
            self.selected_piece = None
