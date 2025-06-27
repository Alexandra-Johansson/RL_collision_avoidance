import numpy as np
import time
import math

from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType

from RLEnv import RLEnv

parameters = {
    'drone_model': DroneModel.CF2X,
    'num_objects': 1,  # Number of balls to add
    'initial_xyzs': np.array([[.0, .0, 1.0]]),
    'ctrl_freq': 60,
    'pyb_freq': 1*240,
    'action_size': 0.2,
    'eval_freq': 10,  # Evaluate every 1 episodes
    'target_pos': np.array([[.0, .0, 1.0]]),
    'target_radius': 0.1,  # Radius of the target sphere
    'avoidance_radius': 1.,  # Radius of the avoidance sphere
    'critical_safety_distance': 0.25,
    'episode_length': 15,  # seconds
    'eval_episodes' : 1,  # Number of episodes to evaluate
    'learning_rate': 3e-4,
    'batch_size': 512, # Should be a divisor of n_steps*n_envs
    'num_steps': 2048, # Number of steps before updating policy, actual update is n_steps*n_envs
    'nr_of_env': 1,  # Number of environments to train in parallel
    'num_epochs': 20,
    'obs_noise': False,  # Add noise to the observations
    'obs_noise_std': 0.01,  # Standard deviation of the noise
    'kf_process_noise': 1e-4,
    'kf_measurement_noise': 1e-2,
    'reward_collision': -5000.0,
    'reward_terminated': 1000.0,
    'reward_target_distance': -100.0,
    'reward_target_distance_delta': 200.0, # Positive for rewarding moving towards target
    'reward_angular_velocity_delta': -5.0, # Negative reward for changing angle
    'reward_object_distance_delta': 100.0, # Positive for rewarding moving away from object
    'reward_step': -100,
    'reward_in_target': 200.0,
    'target_reward': 150000.0,  # Reward to stop training
    'total_timesteps': int(5*1e5),  # Total timesteps to train
}

env = RLEnv(parameters=parameters,gui=True)

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