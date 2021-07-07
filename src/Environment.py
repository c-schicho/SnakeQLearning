"""
Author: Christopher Schicho
Project: Snake Reinforcement Learning
Version: 0.0
"""

import sys
import math
import pygame
import random
from Config import Config as cfg


class Environment:

    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((cfg.width, cfg.height))
        pygame.display.set_caption("Snake Reinforcement Learning by Christopher Schicho")
        self.font = pygame.font.SysFont("comicsansms", 30)

        self.snake = [(cfg.width // 2, cfg.height // 2),
                      (cfg.width // 2 - cfg.grid_size, cfg.height // 2),
                      (cfg.width // 2 - 2 * cfg.grid_size, cfg.height // 2)]
        self.length = 3
        self.direction = cfg.right
        self.food_coordinates = None
        self._spawn_food()
        self.score = 0
        self.reward = 0
        self.moves_since_food = 0
        self.game_over = False
        self.prev_states = None


    ###############
    #### SNAKE ####
    ###############

    def _snake_turn(self, direction):
        """ :param direction:tuple tuple corresponding to the direction """
        # input direction is the same or the opposite of the snake's direction
        if direction == self.direction or (direction[0] * -1, direction[1] * -1) == self.direction:
            return
        # input direction is not the same or the opposite of the snake's direction
        else:
            self.direction = direction

    def _is_game_over(self, new_head):
        """:param new_head:tuple tuple with the coordinates of the new head of the snake """
        # snake crashed in its own body
        if new_head in self.snake:
            self.reward -= 100
            self.game_over = True

        # new head outside the game boundaries
        elif 0 > new_head[0] or cfg.width <= new_head[0] or 0 > new_head[1] or cfg.height <= new_head[1]:
            self.reward -= 100
            self.game_over = True

        # too many senseless moves
        elif self.moves_since_food > len(self.snake) * 20:
            self.reward -= 100
            self.game_over = True

    def _snake_move(self, direction):
        """ :param direction:tuple tuple corresponding to the direction """
        self.reward = 0

        # check for direction change
        self._snake_turn(direction)

        # calculate new head of the snake
        head = self.snake[0]
        new_head = (head[0] + self.direction[0] * cfg.grid_size, head[1] + self.direction[1] * cfg.grid_size)

        self._snake_food_distance(head, new_head)
        self._is_game_over(new_head)

        # game continuing conditions
        if not self.game_over:
            # insert new head on position 0 in snake
            self.snake.insert(0, new_head)
            # check if snake is too long
            if len(self.snake) > self.length:
                self.moves_since_food += 1
                self.snake.pop()

    def _draw_snake(self, surface):
        # draw head of the snake
        head = self.snake[0]
        head_rectangle = pygame.Rect((head[0], head[1]), (cfg.grid_size, cfg.grid_size))
        pygame.draw.rect(surface, cfg.snake_head_color, head_rectangle)

        # draw body of the snake
        for field in self.snake[1:]:
            body_rectangle = pygame.Rect((field[0], field[1]), (cfg.grid_size, cfg.grid_size))
            pygame.draw.rect(surface, cfg.snake_body_color, body_rectangle)

    def is_collision(self, coordinates):
        """ :param coordinates:tuple coordinates to check
            :return collision:bool """
        # snake crashes in its own body
        if coordinates in self.snake:
            return True

        # coordinates outside the boundaries
        elif 0 > coordinates[0] or cfg.width <= coordinates[0] or \
             0 > coordinates[1] or cfg.height <= coordinates[1]:
            return True

        # no collision
        else:
            return False

    def _snake_food_distance(self, head, new_head):
        head_distance = math.sqrt(abs(head[0] - self.food_coordinates[0])**2 +
                                  abs(head[1] - self.food_coordinates[1])**2)

        new_head_distance = math.sqrt(abs(new_head[0] - self.food_coordinates[0])**2 +
                                      abs(new_head[1] - self.food_coordinates[1])**2)

        if new_head_distance <= head_distance:
            self.reward += 1

            if new_head_distance < 2:
                self.reward += 1

        else:
            self.reward -= 1

    def _snake_ate_food(self):
        """ :return snake_ate_food:bool """
        if self.snake[0] == self.food_coordinates:
            self.length += 1
            self.score += 1
            self.reward += 10

            # get new food coordinates
            self._spawn_food()
            return True
        else:
            return False

    def get_snake_state(self):
        """:return snake_state:tuple (snake, length, direction) """
        return (self.snake, self.length, self.direction)


    ##############
    #### FOOD ####
    ##############

    def _spawn_food(self):
        # get random food coordinates
        food_x = random.randint(0, cfg.grid_width - 1) * cfg.grid_size
        food_y = random.randint(0, cfg.grid_height - 1) * cfg.grid_size
        new_food_coordinates = (food_x, food_y)

        # check whether food coordinates are pointing to a free spot
        if new_food_coordinates in self.snake:
            self._spawn_food()
        else:
            self.food_coordinates = new_food_coordinates

    def _draw_food(self, surface):
        """ :param surface:pygame-surface """
        # define food rectangle
        rectangle = pygame.Rect((self.food_coordinates[0], self.food_coordinates[1]), (cfg.grid_size, cfg.grid_size))

        # draw food rectangle
        pygame.draw.rect(surface, cfg.food_color, rectangle)

    def get_food_coordinates(self):
        """ :return food_coordinates:tuple tuple with the food coordinates """
        return self.food_coordinates


    #####################
    #### ENVIRONMENT ####
    #####################

    def reset_environment(self):
        # save old state
        self.prev_states = [self.snake, self.length, self.direction,            # prev state of the snake
                            self.food_coordinates,                              # prev state of the food
                            self.score, self.reward, self.moves_since_food]     # prev state of the environment

        # reset state of the snake
        self.snake = [(cfg.width // 2, cfg.height // 2),
                      (cfg.width // 2 - cfg.grid_size, cfg.height // 2),
                      (cfg.width // 2 - 2 * cfg.grid_size, cfg.height // 2)]
        self.length = 3
        self.direction = cfg.right

        # reset state of the food
        self.food_coordinates = None
        self._spawn_food()

        # reset state of the current game
        self.score = 0
        self.reward = 0
        self.moves_since_food = 0
        self.game_over = False

    def get_game_over_state(self):
        """ :return game_over:bool """
        return self.game_over

    def get_prev_states(self):
        """ :return prev_states:list """
        return self.prev_states

    def environment_step(self, direction):
        """ :param direction:tuple corresponding to the direction
            :return environment_state: reward, score, game_over, ate_food"""

        self.clock.tick(cfg.speed)
        self._check_input()
        self._snake_move(direction)
        ate_food = self._snake_ate_food()
        self.update_window()

        return self.reward, self.score, self.game_over, ate_food


    ################
    #### WINDOW ####
    ################

    def _check_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def update_window(self):
        surface = pygame.Surface(self.window.get_size())
        surface.fill(cfg.surface_color)
        self._draw_snake(surface)
        self._draw_food(surface)
        self.window.blit(surface, (0, 0))
        self.window.blit(self.font.render(f"Score: {self.score}", 1, cfg.font_color), (10, 15))
        pygame.display.update()
