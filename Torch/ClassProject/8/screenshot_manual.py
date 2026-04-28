import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from snake_game import SnakeEnv, draw_game

env = SnakeEnv(board_size=6, seed=42)
screen = pygame.display.set_mode((6*64+40, 6*64+92))
font = pygame.font.SysFont("arial", 26, bold=True)
small_font = pygame.font.SysFont("arial", 16)

for _ in range(20):
    env.step_direction(env.direction)
    if env.done:
        break

draw_game(screen, pygame, env, 64, font, small_font, "Shao")
pygame.image.save(screen, "snake_manual.png")
print("Saved to snake_manual.png")