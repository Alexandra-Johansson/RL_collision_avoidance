import time

import numpy as np
from data_handling import Txt_file
from gym_pybullet_drones.utils.utils import sync
from RLEnv import RLEnv
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

folder = "Training/results/PPO_training_2025.08.08-08.21.46"  # Path to the folder with the trained model

param_file = Txt_file(folder)
parameters = param_file.load_parameters()

model = PPO.load(folder + "/best_model.zip")
print("Model loaded")

parameters['gui'] = True
#test_env = Monitor(RLEnv(parameters = parameters))

test_env = RLEnv(parameters = parameters)
test_env = DummyVecEnv([test_env])
test_env = VecNormalize.load(folder + "/vecnormalize.pkl", test_env)
test_env.training = False
test_env.norm_reward = False

mean_reward, std_reward = evaluate_policy(model,
                                          test_env,
                                          n_eval_episodes=20,
                                          deterministic = True)

print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")