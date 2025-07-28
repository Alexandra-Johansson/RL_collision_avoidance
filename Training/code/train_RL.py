import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel
from train_PPO import Train_PPO

parameters = {
    'drone_model': DroneModel.CF2X,
    'num_objects': 1,  # Number of balls to add
    'initial_xyzs': np.array([[.0, .0, 1.0]]),
    'ctrl_freq': 60,
    'pyb_freq': 1*240,
    'action_size': 0.8,
    'target_pos': np.array([[.0, .0, 1.0]]),
    'target_radius': 0.1,  # Radius of the target sphere
    'avoidance_radius': 2.,  # Radius of the avoidance sphere
    'critical_safety_distance': 0.2,
    'episode_length': 6,  # seconds
    'eval_freq': 20,  # Evaluate every n episodes
    'eval_episodes' : 5,  # Number of episodes to evaluate
    'learning_rate': 5e-4, 
    'batch_size': 512, # Should be a divisor of n_steps*n_envs
    'num_steps': 512, # Number of steps in an episode before updating policy, actual update is n_steps*n_envs
    'nr_of_env': 4,  # Number of environments to train in parallel
    'num_epochs': 5,
    'obs_noise': True,  # Add noise to the observations
    'obs_noise_std': 0.05,  # Standard deviation of the noise
    'kf_process_noise': 1e-4,
    'kf_measurement_noise': 1e-2,
    'reward_collision': -1.0,   # Negative reward for collision
    'reward_sucess': 1.0,   # Positive reward for avoiding collision and returning to target
    'reward_end_outside_target': -0.5,  # Negative reward for ending outside the target
    'reward_truncation': -1.0,  # Negative reward for truncation
    'reward_target_distance': -0.001,
    'reward_target_distance_delta': 0.0, # Positive for rewarding moving towards target
    'reward_rpy': -0.01,  # Negative reward for angular velocity
    'reward_angular_velocity_delta': -5.0, # Negative reward for changing angle
    'reward_object_distance': -0.001,  # Negative reward for being close to the object
    'reward_object_distance_delta': 10.0, # Positive for rewarding moving away from object
    'reward_step': -100,
    'reward_in_target': 200.0,
    'target_reward': 150000.0,  # Reward to stop training
    'total_timesteps': int(4*1e6),  # Total timesteps to train
    'gui': False,  # Whether to use GUI or not
    'obs_timestep': True
}

if __name__ == "__main__":

    time_taken = []

    test_param_action_size = [0.5]
    test_param_learning_rate = [1e-4, 1e-5]
    test_param_reward_target_distance = [-0.1]
    test_param_reward_object_distance_delta = [0.1]
    test_param_reward_in_target = [0.2]
    test_param_reward_object_distance = [0.15, 0.1]
    test_param_reward_rpy = [-0.1]
    test_param_obs_timestep = [False, True]

    for action_size in test_param_action_size:
        parameters['action_size'] = action_size

        for learning_rate in test_param_learning_rate:
            parameters['learning_rate'] = learning_rate

            for reward_target_distance in test_param_reward_target_distance:
                parameters['reward_target_distance'] = reward_target_distance

                for reward_object_distance_delta in test_param_reward_object_distance_delta:
                    parameters['reward_object_distance_delta'] = reward_object_distance_delta

                    for reward_in_target in test_param_reward_in_target:
                        parameters['reward_in_target'] = reward_in_target

                        for reward_object_distance in test_param_reward_object_distance:
                            parameters['reward_object_distance'] = reward_object_distance

                            for reward_rpy in test_param_reward_rpy:
                                parameters['reward_rpy'] = reward_rpy

                                for obs_timestep in test_param_obs_timestep:
                                    parameters['obs_timestep'] = obs_timestep
            
                                    PPO = Train_PPO(parameters=parameters)

                                    time_taken.append(PPO.train())

    print(time_taken)