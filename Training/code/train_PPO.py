import math
import os
import time
from datetime import datetime

import numpy as np
from data_handling import Txt_file, Plot
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

from RLEnv import RLEnv
from data_handling import Plot

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnMaxEpisodes,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
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

        self.filename = os.path.join(self.output_folder, 'PPO_training_' + datetime.now().strftime("%d.%m.%Y-%H.%M.%S"))
        if not os.path.exists(self.filename):
            os.makedirs(self.filename + '/')

        self.plot = Plot(self.filename)

        Param = Txt_file(self.filename)
        Param.save_parameters(self.parameters)

    def train(self):
        # Create the environment
        training_env = make_vec_env(RLEnv, 
                                    env_kwargs = dict(parameters = self.parameters), 
                                    n_envs = self.parameters['nr_of_env'],
                                    seed = 0, 
                                    vec_env_cls=SubprocVecEnv)

        eval_env = SubprocVecEnv([
                                    lambda: Monitor(RLEnv(parameters=self.parameters, gui=self.train_gui))
                                    for _ in range(self.parameters['eval_episodes'])
])

        print("[INFO] Action space: ", training_env.action_space)
        print("[INFO] Observation space: ", training_env.observation_space)

        model = PPO('MultiInputPolicy',
                    training_env,
                    learning_rate = self.parameters['learning_rate'],
                    batch_size = self.parameters['batch_size'],
                    n_epochs = self.parameters['num_epochs'],
                    tensorboard_log = self.filename + '/tensorboard_logs/',
                    verbose = 1)

        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold = self.parameters['target_reward'],
                                                        verbose=1)

        eval_callback = EvalCallback(eval_env,
                                     callback_on_new_best = callback_on_best,
                                     verbose = 1,
                                     n_eval_episodes = self.parameters['eval_episodes'],
                                     best_model_save_path = self.filename,
                                     log_path = self.filename + '/logs/',
                                     eval_freq = max(self.eval_freq // self.parameters['nr_of_env'], 1),
                                     deterministic = True)

        self.plot.tensorboard()

        time_start = time.time()
        model.learn(total_timesteps=self.parameters['total_timesteps'],
                    callback=eval_callback,
                    log_interval = self.parameters['nr_of_env'],
                    progress_bar = True)
        
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

        eval_env.close()
        training_env.close()

    def make_eval_env(self):
        return Monitor(RLEnv(parameters=self.parameters, gui=self.train_gui))