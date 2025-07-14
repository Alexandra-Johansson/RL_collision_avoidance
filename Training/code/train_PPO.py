import math
import os
import time
import traceback
from datetime import datetime

import numpy as np
from data_handling import Plot, Txt_file
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from RLEnv import RLEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnMaxEpisodes,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class Train_PPO():
    def __init__(self, parameters, train_gui : bool = False):
        self.parameters = parameters

        if (self.parameters['num_steps']*self.parameters['nr_of_env'] % self.parameters['batch_size']):
            raise Exception("N_steps*n_envs not divisiable by batch_size")

        self.eval_freq = self.parameters['eval_freq']*self.parameters['num_steps']*self.parameters['nr_of_env']

        self.train_gui = train_gui

        self.output_folder = 'Training/results'

        self.filename = os.path.join(self.output_folder, 'PPO_training_' + datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
        if not os.path.exists(self.filename):
            os.makedirs(self.filename)

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

        # Check environment and outputs warnings
        check_env(training_env)
        
        eval_env = SubprocVecEnv([
                                    lambda: Monitor(RLEnv(parameters=self.parameters, gui=self.train_gui))
                                    for _ in range(self.parameters['eval_episodes'])
                                ])
        #eval_env = DummyVecEnv([lambda: Monitor(RLEnv(parameters=self.parameters, gui=self.train_gui))])

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
                                     eval_freq = self.eval_freq,
                                     deterministic = True)

        self.plot.tensorboard()

        time_start = time.time()
        try:
            model.learn(total_timesteps=self.parameters['total_timesteps'],
                        callback=eval_callback,
                        log_interval = 1,
                        progress_bar = True)
        except Exception as e:
            print("Exception occured: ", e)
            traceback.print_exc()
            input("Error occured! Press enter to continue...")
        time_end = time.time()

        model.save(self.filename+'/final_model.zip')
        print(self.filename)
        print("Model saved")

        time_taken = time_end - time_start
        time_taken_h = math.floor(time_taken/(60*60))
        time_taken -= time_taken_h*60*60
        time_taken_m = math.floor(time_taken/60)
        time_taken_s = round(time_taken - time_taken_m*60)

        time_taken_str = f"Model trained in: {time_taken_h}h.{time_taken_m}m.{time_taken_s}s."

        print(time_taken_str)

        eval_env.close()
        training_env.close()

        return time_taken_str