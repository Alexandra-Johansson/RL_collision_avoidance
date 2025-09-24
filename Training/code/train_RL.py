import numpy as np
from gym_pybullet_drones.utils.enums import ActionType, DroneModel

rl_algorithm = "PPO" # DDPG or PPO

if rl_algorithm == "DDPG":
    from train_DDPG import Train_DDPG

    parameters_DDPG = {
    'drone_model': DroneModel.CF2X,
    'action_type': ActionType.PID,  # Action type for DDPG
    'num_objects': 1,  # Number of balls to add
    'initial_xyzs': np.array([[.0, .0, 1.0]]),
    'ctrl_freq': 60,
    'pyb_freq': 1*240,
    'action_size': 0.5,
    'velocity_size': 3.0,  # Size of the velocity action
    'goal_pos': np.array([[.0, .0, 1.0]]),
    'goal_radius': 0.2,  # Radius of the goal sphere
    'avoidance_radius': 5.,  # Radius of the avoidance sphere
    'critical_safety_distance': 0.2,
    'episode_length': 8,  # seconds
    'eval_freq': 20,  # Evaluate every n episodes
    'eval_episodes' : 5,  # Number of episodes to evaluate
    'learning_rate': 1e-3, 
    'batch_size': 512, # Should be a divisor of n_steps*n_envs
    'num_steps': 1024, # Number of steps in an episode before updating policy, actual update is n_steps*n_envs
    'nr_of_env': 1,  # Number of environments to train in parallel
    'num_epochs': 5,
    'entropy_coefficient': 0.0,
    'obs_noise': False,  # Add noise to the observations
    'obs_noise_std': 0.05,  # Standard deviation of the noise
    'kf_process_noise': 1e-2,
    'kf_measurement_noise': 1e-6,
    'reward_collision': -750.0,   # Negative reward for collision
    'reward_sucess': 1000.0,   # Positive reward for avoiding collision and returning to goal
    'reward_end_outside_goal': -500,  # Negative reward for ending outside the goal
    'reward_truncation': -750.0,  # Negative reward for truncation
    'reward_goal_distance': 10,
    'reward_goal_distance_delta': 0.0, # Positive for rewarding moving towards goal
    'reward_rpy': -0.0,  # Negative reward for angular velocity
    'reward_angular_velocity_delta': -0.0, # Negative reward for changing angle
    'reward_object_distance': -0.0,  # Negative reward for being close to the object
    'reward_object_distance_delta': 0.0, # Positive for rewarding moving away from object
    'reward_action_difference': -1,
    'reward_step': -0,
    'reward_in_goal': 1.0,
    'goal_reward': 150000.0,  # Reward to stop training
    'total_timesteps': int(3*1e6),  # Total timesteps to train
    'space_reduction_transformation': True,
    'gui': False,  # Whether to use GUI or not
    'record': False, # Whether to record simulation, see gym-pybullet-drones git repository
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
    'ctrl_freq': 60,
    'pyb_freq': 1*240,
    'action_size': 0.5,
    'velocity_size': 3.0,
    'goal_pos': np.array([[.0, .0, 1.0]]),
    'goal_radius': 0.1,  # Radius of the goal sphere
    'avoidance_radius': 5.,  # Radius of the avoidance sphere
    'critical_safety_distance': 0.2,
    'episode_length': 4,  # seconds
    'eval_freq': 20,  # Evaluate every n episodes
    'eval_episodes' : 5,  # Number of episodes to evaluate
    'learning_rate': 1e-3, 
    'batch_size': 1024, # Should be a divisor of n_steps*n_envs
    'num_steps': 1024, # Number of steps in an episode before updating policy, actual update is n_steps*n_envs
    'nr_of_env': 1,  # Number of environments to train in parallel, NOT CURRENTLY WORKING for multiple environments
    'num_epochs': 5,
    'entropy_coefficient': 0.0,
    'obs_noise': True,  # Add noise to the observations
    'obs_noise_std': 0.0,  # Standard deviation of the noise
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
    'gui': False,  # Whether to use GUI or not
    'record': False, # Whether to record simulation, see gym-pybullet-drones git repository
    'obs_timestep': False, # Include timestep in observation
    'obs_obj_vel': False, # Include object velocity in observation
    'obs_kf': False, # Include kalmanfilter estimates in observation, otherwise use PyBullet data
    'obs_action_difference': True,  # Include action difference in observation
    'clip_range': 0.2
    }

if __name__ == "__main__":

    time_taken = []

    if rl_algorithm == "PPO":
        test_param_learning_rate = [1e-3]
        test_param_reward_goal_distance = [0.004]
        test_param_reward_object_distance = [-0.2]
        test_param_reward_in_goal = [0]
        test_param_reward_action_difference = [-0.2]
        test_param_clip_range = [0.2]
        test_param_action_size = [0.3]
        test_param_velocity_size = [3.0]
        test_param_ctrl_freq = [30]
        test_param_nr_of_env = [1]
        test_param_action_type = [ActionType.PID]
        test_param_obs_noise_std = [0.0,0.0,0.0]

        for learning_rate in test_param_learning_rate:
            parameters_PPO['learning_rate'] = learning_rate

            for reward_goal_distance in test_param_reward_goal_distance:
                parameters_PPO['reward_goal_distance'] = reward_goal_distance

                for reward_object_distance in test_param_reward_object_distance:
                    parameters_PPO['reward_object_distance'] = reward_object_distance

                    for reward_in_goal in test_param_reward_in_goal:
                        parameters_PPO['reward_in_goal'] = reward_in_goal

                        for reward_action_difference in test_param_reward_action_difference:
                            parameters_PPO['reward_action_difference'] = reward_action_difference

                            for clip_range in test_param_clip_range:
                                parameters_PPO['clip_range'] = clip_range

                                for action_size in test_param_action_size:
                                    parameters_PPO['action_size'] = action_size

                                    for velocity_size in test_param_velocity_size:
                                        parameters_PPO['velocity_size'] = velocity_size

                                        for ctrl_freq in test_param_ctrl_freq:
                                            parameters_PPO['ctrl_freq'] = ctrl_freq

                                            for nr_of_env in test_param_nr_of_env:
                                                parameters_PPO['nr_of_env'] = nr_of_env

                                                for action_type in test_param_action_type:
                                                    parameters_PPO['action_type'] = action_type

                                                    for obs_noise_std in test_param_obs_noise_std:
                                                        parameters_PPO['obs_noise_std'] = obs_noise_std

                                                        PPO = Train_PPO(parameters=parameters_PPO)

                                                        time_taken.append(PPO.train())

    if rl_algorithm == "DDPG":
        test_param_reward_object_distance = [0, -5.0]
        test_param_reward_success = [0, 1000]
        test_param_obs_noise = [True, False]
        test_param_obs_noise_std = [0.05, 0.1, 0.5]

        for reward_object_distance in test_param_reward_object_distance:
            parameters_DDPG["reward_object_distance"] = reward_object_distance

            for reward_sucess in test_param_reward_success:
                parameters_DDPG["reward_sucess"] = reward_sucess

                for obs_noise in test_param_obs_noise:
                    parameters_DDPG["obs_noise"] = obs_noise

                    if obs_noise:
                        for obs_noise_std in test_param_obs_noise_std:
                            parameters_DDPG["obs_noise_std"] = obs_noise_std

                            DDPG = Train_DDPG(parameters = parameters_DDPG)

                            time_taken.append(DDPG.train())

                    else:
                        DDPG = Train_DDPG(parameters = parameters_DDPG)

                        time_taken.append(DDPG.train())
        
    print(time_taken)