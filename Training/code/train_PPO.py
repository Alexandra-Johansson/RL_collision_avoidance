import math
import os
import time
from datetime import datetime

import numpy as np
from data_handling import Txt_file
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from RLEnv import RLEnv
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnMaxEpisodes,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)


class Train_PPO():
    def __init__(self, parameters, train_gui : bool = False):
        self.parameters = parameters

        self.eval_freq = self.parameters['eval_freq']*self.parameters['ctrl_freq']*self.parameters['episode_length']

        self.train_gui = train_gui

        self.output_folder = 'Training/results'

        self.filename = os.path.join(self.output_folder, 'PPO_training_' )#+ datetime.now().strftime("%d.%m.%Y-%H.%M.%S"))
        if not os.path.exists(self.filename):
            os.makedirs(self.filename + '/')

        Param = Txt_file(self.filename)
        Param.save_parameters(self.parameters)

    def train(self):
        # Create the environment
        training_env = make_vec_env(RLEnv, 
                                    env_kwargs = dict(parameters = self.parameters), 
                                    n_envs = self.parameters['nr_of_env'],
                                    seed = 0)

        eval_env = Monitor(RLEnv(parameters=self.parameters, gui=self.train_gui))

        print("[INFO] Action space: ", training_env.action_space)
        print("[INFO] Observation space: ", training_env.observation_space)

        model = PPO('MultiInputPolicy',
                    training_env,
                    learning_rate = self.parameters['learning_rate'],
                    batch_size = self.parameters['batch_size'],
                    verbose = 1)
        
        target_reward = self.parameters['target_reward']

        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)

        eval_callback = EvalCallback(eval_env,
                                     callback_on_new_best = callback_on_best,
                                     verbose = 1,
                                     n_eval_episodes = self.parameters['eval_episodes'],
                                     best_model_save_path = self.filename,
                                     log_path = self.filename + '/logs/',
                                     eval_freq = self.eval_freq,
                                     deterministic = True)
        
        time_start = time.time()
        model.learn(total_timesteps=self.parameters['total_timesteps'],
                    callback=eval_callback,
                    log_interval = self.parameters['nr_of_env'])
        
        time_end = time.time()

        model.save(self.filename+'/final_model.zip')
        print(self.filename)
        print("Model saved")

        time_taken = time_end - time_start
        time_taken_h = math.floor(time_taken/(60*60))
        time_taken -= time_taken_h*60*60
        time_taken_m = math.floor(time_taken/60)
        time_taken_s = round(time_taken - time_taken_m*60)

        print(f"Model trained in: {time_taken_h}h.{time_taken_m}m.{time_taken_s}s.")
        input("Press enter to continue...")

        eval_env.close()
        training_env.close()

        if os.path.isfile(self.filename+'/best_model.zip'):
            path = self.filename+'/best_model.zip'
        else:
            print("[ERROR]: no model under the specified path", self.filename)
        model = PPO.load(path)
        print("Model loaded")

        test_env = RLEnv(parameters = self.parameters, gui = True)

        mean_reward, std_reward = evaluate_policy(model,
                                                    test_env,
                                                    n_eval_episodes=10
                                                    )
        print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")