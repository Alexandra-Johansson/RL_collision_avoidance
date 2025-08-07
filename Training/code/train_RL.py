import numpy as np
from gym_pybullet_drones.utils.enums import ActionType, DroneModel

rl_algorithm = "DDPG" # DDPG or PPO
if rl_algorithm == "DDPG":
    from train_DDPG import Train_DDPG

    parameters_DDPG = {
    'drone_model': DroneModel.CF2X,
    'num_objects': 1,  # Number of balls to add
    'initial_xyzs': np.array([[.0, .0, 1.0]]),
    'ctrl_freq': 60,
    'pyb_freq': 1*240,
    'action_size': 0.5,
    'target_pos': np.array([[.0, .0, 1.0]]),
    'target_radius': 0.2,  # Radius of the target sphere
    'avoidance_radius': 5.,  # Radius of the avoidance sphere
    'critical_safety_distance': 0.2,
    'episode_length': 8,  # seconds
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
    'reward_target_distance': -0.0,
    'reward_target_distance_delta': 0.0, # Positive for rewarding moving towards target
    'reward_rpy': -0.0,  # Negative reward for angular velocity
    'reward_angular_velocity_delta': -0.0, # Negative reward for changing angle
    'reward_object_distance': -0.0,  # Negative reward for being close to the object
    'reward_object_distance_delta': 0.0, # Positive for rewarding moving away from object
    'reward_action_difference': -0.15,
    'reward_step': -0,
    'reward_in_target': 0.0,
    'target_reward': 150000.0,  # Reward to stop training
    'total_timesteps': int(3*1e6),  # Total timesteps to train
    'gui': False,  # Whether to use GUI or not
    'obs_timestep': False, # Include timestep in observation
    'obs_obj_vel': False, # Include object velocity in observation
    'obs_kf': False, # Include kalmanfilter estimates in observation, otherwise use PyBullet data
    'obs_action_difference': True,  # Include action difference in observation
    'learning_starts': 100000,
    'training_freq': 1, # Update model every 'train_freq' steps
    'gradient_steps': -1 # Amount of gradient steps, -1 = same as train_freq
    }

if rl_algorithm == "PPO":
    from train_PPO import Train_PPO

    parameters_PPO = {
    'drone_model': DroneModel.CF2X,
    'action_type': ActionType.PID,
    'num_objects': 1,  # Number of balls to add
    'initial_xyzs': np.array([[.0, .0, 1.0]]),
    'ctrl_freq': 30,
    'pyb_freq': 1*240,
    'action_size': 0.5,
    'target_pos': np.array([[.0, .0, 1.0]]),
    'target_radius': 0.2,  # Radius of the target sphere
    'avoidance_radius': 5.,  # Radius of the avoidance sphere
    'critical_safety_distance': 0.2,
    'episode_length': 8,  # seconds
    'eval_freq': 20,  # Evaluate every n episodes
    'eval_episodes' : 5,  # Number of episodes to evaluate
    'learning_rate': 5e-4, 
    'batch_size': 1024, # Should be a divisor of n_steps*n_envs
    'num_steps': 1024, # Number of steps in an episode before updating policy, actual update is n_steps*n_envs
    'nr_of_env': 1,  # Number of environments to train in parallel
    'num_epochs': 5,
    'entropy_coefficient': 0.0,
    'obs_noise': False,  # Add noise to the observations
    'obs_noise_std': 0.05,  # Standard deviation of the noise
    'kf_process_noise': 1e-2,
    'kf_measurement_noise': 1e-6,
    'reward_collision': -1.0,   # Negative reward for collision
    'reward_sucess': 1.0,   # Positive reward for avoiding collision and returning to target
    'reward_end_outside_target': -0.75,  # Negative reward for ending outside the target
    'reward_truncation': -1.0,  # Negative reward for truncation
    'reward_target_distance': -0.0,
    'reward_target_distance_delta': 0.0, # Positive for rewarding moving towards target
    'reward_rpy': -0.0,  # Negative reward for angular velocity
    'reward_angular_velocity_delta': -0.0, # Negative reward for changing angle
    'reward_object_distance': -0.0,  # Negative reward for being close to the object
    'reward_object_distance_delta': 0.0, # Positive for rewarding moving away from object
    'reward_step': -0,
    'reward_in_target': 0.0,
    'target_reward': 150000.0,  # Reward to stop training
    'total_timesteps': int(6*1e6),  # Total timesteps to train
    'gui': False,  # Whether to use GUI or not
    'obs_timestep': False, # Include timestep in observation
    'obs_obj_vel': True, # Include object velocity in observation
    'obs_kf': False, # Include kalmanfilter estimates in observation, otherwise use PyBullet data
    'obs_action_difference': True,  # Include action difference in observation
    'clip_range': 0.2
    }

if __name__ == "__main__":

    time_taken = []

    if rl_algorithm == "PPO":
        test_param_learning_rate = [5e-4]
        test_param_reward_target_distance = [-0.1]
        test_param_obs_action_difference = [True]
        test_param_clip_range = [0.2]
        test_param_action_size = [0.3]
        test_param_ctrl_freq = [30]

        for learning_rate in test_param_learning_rate:
            parameters_PPO['learning_rate'] = learning_rate

            for reward_target_distance in test_param_reward_target_distance:
                parameters_PPO['reward_target_distance'] = reward_target_distance

                for obs_action_difference in test_param_obs_action_difference:
                    parameters_PPO['obs_action_difference'] = obs_action_difference

                    for clip_range in test_param_clip_range:
                        parameters_PPO['clip_range'] = clip_range

                        for action_size in test_param_action_size:
                            parameters_PPO['action_size'] = action_size

                            for ctrl_freq in test_param_ctrl_freq:
                                parameters_PPO['ctrl_freq'] = ctrl_freq

                                PPO = Train_PPO(parameters=parameters_PPO)

                                time_taken.append(PPO.train())

    if rl_algorithm == "DDPG":
        test_param_reward_object_distance = [-0.5, -1.0, -1.5]
        test_param_nr_of_env = [4, 1]

        for reward_object_distance in test_param_reward_object_distance:
            parameters_DDPG["reward_object_distance"] = reward_object_distance

            for nr_of_env in test_param_nr_of_env:
                parameters_DDPG["nr_of_env"] = nr_of_env

                DDPG = Train_DDPG(parameters = parameters_DDPG)

                time_taken.append(DDPG.train())
        
    print(time_taken)