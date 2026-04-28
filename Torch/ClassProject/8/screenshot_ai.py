import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from snake_game import SnakeEnv, draw_game

q_table = np.load("snake_rl_outputs/q_table.npy")

env = SnakeEnv(board_size=6, seed=2024)
screen = pygame.display.set_mode((6*64+40, 6*64+92))
font = pygame.font.SysFont("arial", 26, bold=True)
small_font = pygame.font.SysFont("arial", 16)

for _ in range(15):
    state_id = env.get_state_id()
    action_id = int(np.argmax(q_table[state_id]))
    env.step_relative(action_id)
    if env.done:
        break

draw_game(screen, pygame, env, 64, font, small_font, "Shao")
pygame.image.save(screen, "snake_ai.png")
print("Saved to snake_ai.png")