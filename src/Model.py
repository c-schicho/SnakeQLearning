"""
Author: Christopher Schicho
Project: Snake Reinforcement Learning
Version: 0.0
"""

import os
import torch
from Config import Config as cfg


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model_path = "./model"
        self.model = torch.nn.Sequential(
            torch.nn.Linear(cfg.input_neurons, cfg.hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.hidden_neurons, cfg.output_neurons))

    def forward(self, model_input):
        return self.model(model_input)

    def save_model(self, file_name="snake_model.pth"):
        try:
            os.makedirs(self.model_path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.model_path, file_name))
            print("Model successfully saved\n")
        except:
            print("Model could not be saved\n")

    def load_model(self, file_name="snake_model.pth"):
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.model_path, file_name)))
            print("Model successfully loaded\n")
        except FileNotFoundError:
            print("Model has not been loaded: file not found\n")


class Trainer:

    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=cfg.l_rate) #Adam
        self.loss_function = torch.nn.MSELoss()

    def train_step(self, prev_state, state, direction_arr, reward, game_over):
        prev_state = torch.tensor(prev_state, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)
        directions = torch.tensor(direction_arr, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            prev_state = torch.unsqueeze(prev_state, 0)
            state = torch.unsqueeze(state, 0)
            directions = torch.unsqueeze(directions, 0)
            reward = torch.unsqueeze(reward, 0)

        prediction = self.model.forward(prev_state)

        target = prediction.clone()
        q_new = reward[0]

        if not game_over:
            q_new = reward[0] + cfg.gamma * torch.max(self.model.forward(state[0]))

        target[0][torch.argmax(directions[0]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.loss_function(target, prediction)
        loss.backward()
        self.optimizer.step()
