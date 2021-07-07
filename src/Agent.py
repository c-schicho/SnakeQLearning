"""
Author: Christopher Schicho
Project: Snake Reinforcement Learning
Version: 0.0
"""

import torch
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from Environment import Environment
from Config import Config as cfg
from Model import Model, Trainer


class Agent:

    def __init__(self):
        self.n_games = 0
        self.memory = deque(maxlen=cfg.max_mem)
        self.epsilon = cfg.epsilon
        self.epsilon_discount = cfg.epsilon_discount
        self.gamma = cfg.gamma
        self.model = Model()
        self.trainer = Trainer(self.model)

    def get_state(self, env):
        """ :param env:Environment object
            :return state:list current state of the game step """
        # get environment states
        snake = env.get_snake_state()
        direction = snake[2]
        food = env.get_food_coordinates()

        # simulate the head in each direction
        head = snake[0][0]
        head_left = (head[0] - cfg.grid_size, head[1])
        head_right = (head[0] + cfg.grid_size, head[1])
        head_up = (head[0], head[1] - cfg.grid_size)
        head_down = (head[0], head[1] + cfg.grid_size)

        state = np.array([
            # move direction left
            direction == cfg.left,
            # move direction right
            direction == cfg.right,
            # move direction up
            direction == cfg.up,
            # move direction down
            direction == cfg.down,
            # danger left
            env.is_collision(head_left),
            # danger right
            env.is_collision(head_right),
            # danger above
            env.is_collision(head_up),
            # danger below
            env.is_collision(head_down),
            # food location left
            head[0] > food[0],
            # food location right
            head[0] < food[0],
            # food location up
            head[1] > food[1],
            # food location down
            head[1] < food[1]
            ], dtype=np.int)

        return state

    def get_direction(self, state):
        """ :param state:list current state of the game step
            :return direction:tuple next direction to move to """

        directions = [cfg.left, cfg.right, cfg.up, cfg.down]

        # exploration (only during training)
        if random.uniform(0, 1) < self.epsilon and cfg.train_agent and not cfg.load_model:
            self.epsilon *= self.epsilon_discount
            index = random.randint(0, 3)
            new_direction = directions[index]

        # deploy learned
        else:
            state = torch.tensor(state, dtype=torch.float)
            prediction = self.model.forward(state)
            _, index = prediction.max(-1)
            new_direction = directions[index]

        direction_arr = np.array([0, 0, 0, 0], dtype=np.float)
        direction_arr[index] = 1

        return new_direction, direction_arr

    def train_shortterm_memory(self, prev_state, state, direction, reward, game_over):
        self.trainer.train_step(prev_state, state, direction, reward, game_over)

    def train_longtrem_memory(self):
        if len(self.memory) > cfg.batch_size:
            sample = random.sample(self.memory, cfg.batch_size)

        else:
            sample = self.memory

        prev_states, states, directions, rewards, game_over = zip(*sample)
        self.trainer.train_step(prev_states, states, directions, rewards, game_over)

    def save_agent_state(self, prev_state, state, direction, reward, game_over):
        self.memory.append((prev_state, state, direction, reward, game_over))



#######################
#### EXECUTE AGENT ####
#######################


def execute_agent():
    env = Environment()
    agent = Agent()
    record = 0

    if cfg.load_model:
        agent.model.load_model()

    if cfg.train_agent:
        scores = pd.DataFrame(columns=["score", "mean_score"])

        while agent.n_games < cfg.train_iterations:
            prev_state = agent.get_state(env)
            new_direction, new_direction_arr = agent.get_direction(prev_state)
            reward, score, game_over, _ = env.environment_step(new_direction)

            new_state = agent.get_state(env)
            agent.train_shortterm_memory(prev_state, new_state, new_direction_arr, reward, game_over)
            agent.save_agent_state(prev_state, new_state, new_direction_arr, reward, game_over)

            if game_over:
                env.reset_environment()
                agent.n_games += 1
                record = max(score, record)

                agent.train_longtrem_memory()

                print(f"Game: {agent.n_games:6d}\t Epsilon: {agent.epsilon:.3f}\t Score: {score}\t Record: {record}")

                if cfg.plot_training:
                    # plot the trainings progress
                    mean_score = (scores["score"].sum() + score) / (len(scores["score"]) + 1)
                    scores = scores.append({"score": score, "mean_score": mean_score}, ignore_index=True)

                    plt.clf()
                    plt.get_current_fig_manager().canvas.set_window_title("Snake Reinforcement Learning by Christopher Schicho")
                    plt.title(f"Training     Record: {record}     Mean Score: {mean_score:.3f}\n")
                    plt.xlabel("Number of Games")
                    plt.ylabel("Score\n")
                    plt.plot(scores["score"], label="Game Score")
                    plt.plot(scores["mean_score"], label="Mean Score")
                    plt.xlim(xmin=0)
                    plt.ylim(ymin=0, ymax=record+5)
                    plt.grid(axis="y", alpha=0.5)
                    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)
                    plt.pause(0.01)

        plt.show()

    # execute agent without training
    else:
        while True:
            state = agent.get_state(env)
            new_direction, _ = agent.get_direction(state)
            _, score, game_over, _ = env.environment_step(new_direction)

            if game_over:
                env.reset_environment()
                agent.n_games += 1
                record = max(score, record)

                print(f"Game: {agent.n_games:6d}\t Score: {score}\t Record: {record}")

    if cfg.save_model:
        agent.model.save_model()



if __name__ == "__main__":
    cfg().print_parameters()

    if cfg.plot_training:
        matplotlib.use('TKAgg', force=True)

    execute_agent()