import time

import numpy as np
from data_handling import Txt_file
from gym_pybullet_drones.utils.utils import sync
from RLEnv import RLEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

folder = "Training/results/PPO_training_2025.08.05-23.12.00"  # Path to the folder with the trained model

param_file = Txt_file(folder)
parameters = param_file.load_parameters()

model = PPO.load(folder + "/final_model.zip")
print("Model loaded")

input("Press enter to continue...")

parameters['gui'] = True
test_env = Monitor(RLEnv(parameters = parameters))

mean_reward, std_reward = evaluate_policy(model,
                                          test_env,
                                          n_eval_episodes=10)
print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")