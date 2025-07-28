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
    BaseCallback,
    CallbackList,
    EvalCallback,
    StopTrainingOnMaxEpisodes,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_infos = []

    def _on_step(self) -> bool:
        # Check if an episode ended
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos,dones):
            if done and info is not None:
                self.episode_infos.append(info)
        return True

    def _on_rollout_end(self):
        # Log all episode infos collected during the rollout
        min_obj_distances = [
            info["min_object_distance"]
            for info in self.episode_infos
            if info is not None and "min_object_distance" in info
        ]
        if min_obj_distances:
            mean_min_obj_distance = np.mean(min_obj_distances)
            self.logger.record("custom/mean_min_object_distance", mean_min_obj_distance)

        max_target_distances = [
            info["max_target_distance"]
            for info in self.episode_infos
            if info is not None and "max_target_distance" in info
        ]
        if max_target_distances:
            mean_max_target_distance = np.mean(max_target_distances)
            self.logger.record("custom/mean_max_target_distance", mean_max_target_distance)

        final_drone_altitudes = [
            info["final_drone_altitude"]
            for info in self.episode_infos
            if info is not None and "final_drone_altitude" in info
        ]
        if final_drone_altitudes:
            mean_final_drone_altitude = np.mean(final_drone_altitudes)
            self.logger.record("custom/mean_final_drone_altitude", mean_final_drone_altitude)

        final_target_distances = [
            info["final_target_distance"]
            for info in self.episode_infos
            if info is not None and "final_target_distance" in info
        ]
        if final_target_distances:
            mean_final_target_distance = np.mean(final_target_distances)
            self.logger.record("custom/mean_final_target_distance", mean_final_target_distance)

        out_of_bounds = [
            info["out_of_bounds"]
            for info in self.episode_infos
            if info is not None and "out_of_bounds" in info
        ]
        if out_of_bounds:
            mean_out_of_bounds = np.mean([int(ob) for ob in out_of_bounds])
            self.logger.record("custom/mean_out_of_bounds", mean_out_of_bounds)

        orientation_out_of_bounds = [
            info["orientation_out_of_bounds"]
            for info in self.episode_infos
            if info is not None and "orientation_out_of_bounds" in info
        ]
        if orientation_out_of_bounds:
            mean_orientation_out_of_bounds = np.mean([int(oob) for oob in orientation_out_of_bounds])
            self.logger.record("custom/mean_orientation_out_of_bounds", mean_orientation_out_of_bounds)

        time_limit_reached = [
            info["time_limit_reached"]
            for info in self.episode_infos
            if info is not None and "time_limit_reached" in info
        ]
        if time_limit_reached:
            mean_time_limit_reached = np.mean([int(tlr) for tlr in time_limit_reached])
            self.logger.record("custom/mean_time_limit_reached", mean_time_limit_reached)

        object_collisions = [
            info["obj_collision"]
            for info in self.episode_infos
            if info is not None and "obj_collision" in info
        ]
        if object_collisions:
            mean_object_collision = np.mean([int(oc) for oc in object_collisions])
            self.logger.record("custom/mean_object_collision", mean_object_collision)

        contact_collisions = [
            info["contact_collision"]
            for info in self.episode_infos
            if info is not None and "contact_collision" in info
        ]
        if contact_collisions:
            mean_contact_collision = np.mean([int(cc) for cc in contact_collisions])
            self.logger.record("custom/mean_contact_collision", mean_contact_collision)
        
        self.episode_infos = [] # Resetting episode/rollout info

        return True
    
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
        
        training_env = VecNormalize(training_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        self.parameters['gui'] = False
        eval_env = SubprocVecEnv([
                                    lambda: Monitor(RLEnv(parameters=self.parameters))
                                    for _ in range(self.parameters['eval_episodes'])
                                ])
        
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.obs_rms = training_env.obs_rms  # Use the same normalization as the training environment

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