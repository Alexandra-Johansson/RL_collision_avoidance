import time
import numpy as np

from gym_pybullet_drones.utils.utils import sync

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from RLEnv import RLEnv
from data_handling import Txt_file

folder = "Training/results/PPO_training_24.06.2025-10.44.29"  # Path to the folder with the trained model

param_file = Txt_file(folder)
parameters = param_file.load_parameters()

model = PPO.load(folder + "/best_model.zip")
print("Model loaded")

test_env = Monitor(RLEnv(parameters = parameters, gui = True))

mean_reward, std_reward = evaluate_policy(model,
                                          test_env,
                                          n_eval_episodes=10)
print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")