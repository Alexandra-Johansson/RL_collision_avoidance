import time

import numpy as np
from data_handling import Txt_file
from gym_pybullet_drones.utils.utils import sync
from RLEnv import RLEnv
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import pybullet as pb

folder = "final_models/transformations/PPO_transformation_true_7"  # Path to the folder with the trained model

param_file = Txt_file(folder)
parameters = param_file.load_parameters()

model = PPO.load(folder + "/final_model.zip")
print("Model loaded")

parameters['gui'] = True
parameters['record'] = True

test_env = DummyVecEnv([lambda: RLEnv(parameters = parameters)])
test_env = VecNormalize.load(folder + "/final_vecnormalize.pkl", test_env)
test_env.training = False
test_env.norm_reward = False


# Top-down view of the simulation
pb.resetDebugVisualizerCamera(
    cameraDistance=3.0,     # Distance from the target
    cameraYaw=30,           # Horizontal angle (left/right)
    cameraPitch=-90,        # Vertical angle (up/down)
    cameraTargetPosition=[0, 0, 1]  # Where the camera is looking
)


#input("Press enter...")

mean_reward, std_reward = evaluate_policy(model,
                                          test_env,
                                          n_eval_episodes=10,
                                          deterministic = True)

print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")