import math
import os
import time
import traceback
from datetime import datetime

import numpy as np
from data_handling import Plot, Txt_file
from RLEnv import RLEnv
from custom_callbacks import CustomTensorboardCallback, SaveVecNormalizeCallback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

class Train_PPO():
    def __init__(self, parameters):
        self.parameters = parameters

        if (self.parameters['num_steps']*self.parameters['nr_of_env'] % self.parameters['batch_size']):
            raise Exception("N_steps*n_envs not divisiable by batch_size")

        self.eval_freq = self.parameters['eval_freq']*self.parameters['num_steps']

        self.output_folder = 'Training/results'

        self.filename = os.path.join(self.output_folder, 'PPO_training_' + datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
        if not os.path.exists(self.filename):
            os.makedirs(self.filename)

        self.plot = Plot(self.filename)

        Param = Txt_file(self.filename)
        Param.save_parameters(self.parameters)

    def train(self):
        # Check environment and outputs warnings
        test_parameters = self.parameters.copy()
        test_parameters['gui'] = False  # Disable GUI for environment check
        test_env = RLEnv(parameters=self.parameters)
        check_env(test_env)

        # Create the environment
        training_env = make_vec_env(RLEnv, 
                                    env_kwargs = dict(parameters = self.parameters), 
                                    n_envs = self.parameters['nr_of_env'],
                                    seed = 0, 
                                    vec_env_cls=SubprocVecEnv)
        
        training_env = VecNormalize(training_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        self.parameters['gui'] = False
        eval_env = SubprocVecEnv([
                                    lambda: Monitor(RLEnv(parameters=self.parameters))
                                    for _ in range(self.parameters['eval_episodes'])
                                ])
        
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env.obs_rms = training_env.obs_rms  # Use the same normalization as the training environment

        print("[INFO] Action space: ", training_env.action_space)
        print("[INFO] Observation space: ", training_env.observation_space)

        model = PPO('MultiInputPolicy',
                    training_env,
                    learning_rate = self.parameters['learning_rate'],
                    batch_size = self.parameters['batch_size'],
                    n_epochs = self.parameters['num_epochs'],
                    clip_range = self.parameters['clip_range'],
                    ent_coef = self.parameters['entropy_coefficient'],
                    tensorboard_log = self.filename + '/tensorboard_logs/',
                    verbose = 1)

        save_vec_norm_callback = SaveVecNormalizeCallback(vec_env = eval_env,
                                                          save_dir = self.filename,
                                                          verbose = 1)

        eval_callback = EvalCallback(eval_env,
                                     callback_on_new_best = save_vec_norm_callback,
                                     verbose = 1,
                                     n_eval_episodes = self.parameters['eval_episodes'],
                                     eval_freq = self.eval_freq,
                                     log_path = self.filename + '/logs/',
                                     best_model_save_path = self.filename,
                                     deterministic = True)

        callback_list = CallbackList([eval_callback, CustomTensorboardCallback()])

        self.plot.tensorboard()

        time_start = time.time()
        try:
            model.learn(total_timesteps=self.parameters['total_timesteps'],
                        callback=callback_list,
                        log_interval = 1,
                        progress_bar = True)
        except Exception as e:
            print("Exception occured: ", e)
            traceback.print_exc()
            input("Error occured! Press enter to continue...")
        time_end = time.time()

        model.save(self.filename+'/final_model.zip')
        training_env.save(self.filename + "/final_vecnormalize.pkl")
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