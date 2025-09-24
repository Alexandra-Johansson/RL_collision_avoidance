import math
import time

import numpy as np
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType
from RLEnv import RLEnv

parameters = {
    'drone_model': DroneModel.CF2X,
    'action_type': ActionType.PID,
    'num_objects': 1,  # Number of balls to add
    'initial_xyzs': np.array([[.0, .0, 1.0]]),
    'ctrl_freq': 60,
    'pyb_freq': 1*240,
    'action_size': 0.5,
    'velocity_size': 3.0,
    'goal_pos': np.array([[.0, .0, 1.0]]),
    'goal_radius': 0.1,  # Radius of the goal sphere
    'avoidance_radius': 5.,  # Radius of the avoidance sphere
    'critical_safety_distance': 0.2,
    'episode_length': 3,  # seconds
    'eval_freq': 20,  # Evaluate every n episodes
    'eval_episodes' : 5,  # Number of episodes to evaluate
    'learning_rate': 1e-3, 
    'batch_size': 1024, # Should be a divisor of n_steps*n_envs
    'num_steps': 1024, # Number of steps in an episode before updating policy, actual update is n_steps*n_envs
    'nr_of_env': 1,  # Number of environments to train in parallel, NOT CURRENTLY WORKING for multiple environments
    'num_epochs': 5,
    'entropy_coefficient': 0.0,
    'obs_noise': True,  # Add noise to the observations
    'obs_noise_std': 0.5,  # Standard deviation of the noise
    'kf_process_noise': 1e-2,
    'kf_measurement_noise': 1e-6,
    'reward_collision': -1.0,   # Negative reward for collision
    'reward_sucess': 1.0,   # Positive reward for avoiding collision and returning to goal
    'reward_end_outside_goal': -0.75,  # Negative reward for ending outside the goal
    'reward_truncation': -1.0,  # Negative reward for truncation
    'reward_goal_distance': 0.003,
    'reward_goal_distance_delta': 0.0, # Positive for rewarding moving towards goal
    'reward_rpy': -0.0,  # Negative reward for angular velocity
    'reward_angular_velocity_delta': -0.0, # Negative reward for changing angle
    'reward_object_distance': -0.0,  # Negative reward for being close to the object
    'reward_object_distance_delta': 0.0, # Positive for rewarding moving away from object
    'reward_action_difference': -0.1,
    'reward_step': -0,
    'reward_in_goal': 0.25,
    'goal_reward': 150000.0,  # Reward to stop training
    'total_timesteps': int(2*1e6),  # Total timesteps to train
    'space_reduction_transformation': True,
    'gui': True,  # Whether to use GUI or not
    'record': False, # Whether to record simulation, see gym-pybullet-drones git repository
    'obs_timestep': False, # Include timestep in observation
    'obs_obj_vel': False, # Include object velocity in observation
    'obs_kf': False, # Include kalmanfilter estimates in observation, otherwise use PyBullet data
    'obs_action_difference': True,  # Include action difference in observation
    'clip_range': 0.2
    }

env = RLEnv(parameters=parameters)

action = np.array([[.0, .0, .0]])

for i in range (10000):
        '''
        x = math.cos(4*2*math.pi*i/10000)
        y = math.sin(4*2*math.pi*i/10000)
        action[0,0] = x
        action[0,1] = y
        '''
        if (i%100 == 0):
            # Add a ball at a random position
            env.addBallRandom()

            # Add a ball at a fixed position for collision testing
            #env.addBall([0, 0, 0.5], [0, 0, 100])

            #env.reset()
            pass
        

        obs, reward, terminated, truncated, info = env.step(action)
        #print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated, "\tInfo:", info)
        #time.sleep(1./240.)
        #time.sleep(1./10.)