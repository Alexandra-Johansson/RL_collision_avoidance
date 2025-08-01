import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel
from train_PPO import Train_PPO

parameters = {
    'drone_model': DroneModel.CF2X,
    'num_objects': 1,  # Number of balls to add
    'initial_xyzs': np.array([[.0, .0, 1.0]]),
    'ctrl_freq': 60,
    'pyb_freq': 1*240,
    'action_size': 0.5,
    'target_pos': np.array([[.0, .0, 1.0]]),
    'target_radius': 0.1,  # Radius of the target sphere
    'avoidance_radius': 2.,  # Radius of the avoidance sphere
    'critical_safety_distance': 0.2,
    'episode_length': 6,  # seconds
    'eval_freq': 20,  # Evaluate every n episodes
    'eval_episodes' : 5,  # Number of episodes to evaluate
    'learning_rate': 5e-4, 
    'batch_size': 1024, # Should be a divisor of n_steps*n_envs
    'num_steps': 1024, # Number of steps in an episode before updating policy, actual update is n_steps*n_envs
    'nr_of_env': 4,  # Number of environments to train in parallel
    'num_epochs': 5,
    'entropy_coefficient': 0.0,
    'obs_noise': False,  # Add noise to the observations
    'obs_noise_std': 0.05,  # Standard deviation of the noise
    'kf_process_noise': 1e-2,
    'kf_measurement_noise': 1e-6,
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
    'total_timesteps': int(3*1e6),  # Total timesteps to train
    'gui': False,  # Whether to use GUI or not
    'obs_timestep': False, # Include timestep in observation
    'obs_obj_vel': True, # Include object velocity in observation
    'obs_kf': False, # Include kalmanfilter estimates in observation, otherwise use PyBullet data
    'obs_action_difference': True,  # Include action difference in observation
    'clip_range': 0.2
}

if __name__ == "__main__":

    time_taken = []

    test_param_learning_rate = [7e-4]
    test_param_reward_target_distance = [-0.1]
    test_param_reward_object_distance_delta = [0.1]
    test_param_reward_in_target = [0.2]
    test_param_reward_object_distance = [0.15]
    test_param_reward_rpy = [-0.15]
    test_param_obs_timestep = [False]
    test_param_obs_action_difference = [False]
    test_param_entropy_coefficent = [0, 10]
    test_param_reward_truncation = [-1.0, -0.5]
    test_param_clip_range = [0.3, 0.5]

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

                                for reward_truncation in test_param_reward_truncation:
                                    parameters['reward_truncation'] = reward_truncation

                                    for obs_action_difference in test_param_obs_action_difference:
                                        parameters['obs_action_difference'] = obs_action_difference

                                        for entropy_coefficient in test_param_entropy_coefficent:
                                            parameters['entropy_coefficient'] = entropy_coefficient

                                            for clip_range in test_param_clip_range:
                                                parameters['clip_range'] = clip_range
            
                                                PPO = Train_PPO(parameters=parameters)

                                                time_taken.append(PPO.train())

    print(time_taken)