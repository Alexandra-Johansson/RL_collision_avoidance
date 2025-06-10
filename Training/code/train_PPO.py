import os
import time
import numpy as np
from datetime import datetime

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnMaxEpisodes
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.buffers import DictReplayBuffer

from stable_baselines3.common.monitor import Monitor    

from RLEnv import RLEnv
from data_handling import Txt_file

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
                                     best_model_save_path = self.filename + '/best_model/',
                                     log_path = self.filename + '/logs/',
                                     eval_freq = self.eval_freq,
                                     deterministic = True)
        
        model.learn(total_timesteps=self.parameters['total_timesteps'],
                    callback=eval_callback,
                    log_interval = self.parameters['nr_of_env'])