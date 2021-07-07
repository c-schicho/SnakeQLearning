"""
Author: Christopher Schicho
Project: Snake Reinforcement Learning
Version: 0.0
"""


class Config:
    #################
    #### GENERAL ####
    #################

    train_agent = True
    plot_training = True

    load_model = False
    save_model = True

    #####################
    #### ENVIRONMENT ####
    #####################

    # window
    width = 660    #22 * 30 (grid_size)
    height = 480   #16 * 30 (grid_size)

    # speed
    speed = 200 #80

    # game surface
    grid_size = 30
    grid_width = width // grid_size
    grid_height = height // grid_size

    # game colors
    surface_color = (70, 70, 70)
    snake_head_color = (20, 200, 20)
    snake_body_color = (20, 170, 20)
    food_color = (200, 150, 50)
    font_color = (255, 255, 255)

    # directions
    left = (-1, 0)
    right = (1, 0)
    up = (0, -1)
    down = (0, 1)

    ###############
    #### AGENT ####
    ###############

    # exploration rate
    epsilon = 1.0
    epsilon_discount = 0.99

    # maximum memory
    max_mem = 50_000

    ###############
    #### MODEL ####
    ###############

    # layers
    input_neurons = 12
    hidden_neurons = 256
    output_neurons = 4

    # iterations
    train_iterations = 400

    # batch size
    batch_size = 20

    # learning rate
    l_rate = 0.005

    #
    gamma = 0.7

    def __init__(self):
        pass

    def print_parameters(self):
        print(f"\ntrain_agent: {self.train_agent}")
        print(f"plot_training: {self.plot_training}")
        print(f"load_model: {self.load_model}")
        print(f"save_model: {self.save_model}\n")

        print(f"train_iterations: {self.train_iterations}")
        print(f"batch_size: {self.batch_size}")
        print(f"learning_rate: {self.l_rate}")
        print(f"gamma: {self.gamma}")
        print(f"epsilon: {self.epsilon}")
        print(f"epsilon_discount: {self.epsilon_discount}\n")
