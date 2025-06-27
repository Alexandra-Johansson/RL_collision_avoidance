import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel
from train_PPO import Train_PPO

parameters = {
    'drone_model': DroneModel.CF2X,
    'num_objects': 1,  # Number of balls to add
    'initial_xyzs': np.array([[.0, .0, 1.0]]),
    'ctrl_freq': 60,
    'pyb_freq': 1*240,
    'action_size': 0.2,
    'eval_freq': 10,  # Evaluate every 1 episodes
    'target_pos': np.array([[.0, .0, 1.0]]),
    'target_radius': 0.05,  # Radius of the target sphere
    'avoidance_radius': 1.,  # Radius of the avoidance sphere
    'critical_safety_distance': 0.25,
    'episode_length': 15,  # seconds
    'eval_episodes' : 1,  # Number of episodes to evaluate
    'learning_rate': 3e-3,
    'batch_size': 512, # Should be a divisor of n_steps*n_envs
    'num_steps': 2048, # Number of steps before updating policy, actual update is n_steps*n_envs
    'nr_of_env': 1,  # Number of environments to train in parallel, Multiple environments is not currently implemented correctly
    'num_epochs': 20,
    'obs_noise': True,  # Add noise to the observations
    'obs_noise_std': 0.05,  # Standard deviation of the noise
    'kf_process_noise': 1e-4,
    'kf_measurement_noise': 1e-2,
    'reward_collision': -5000.0,
    'reward_terminated': 1000.0,
    'reward_target_distance': -100.0,
    'reward_target_distance_delta': 200.0, # Positive for rewarding moving towards target
    'reward_angular_velocity_delta': -5.0, # Negative reward for changing angle
    'reward_object_distance_delta': 10.0, # Positive for rewarding moving away from object
    'reward_step': -50,
    'reward_in_target': 100.0,
    'target_reward': 150000.0,  # Reward to stop training
    'total_timesteps': int(5*1e5),  # Total timesteps to train
}

if __name__ == "__main__":

    test_param_nr_of_env = [1]
    time_taken = []

    for nr_of_env in test_param_nr_of_env:
        parameters['nr_of_env'] = nr_of_env

        PPO = Train_PPO(parameters=parameters, train_gui=False)

        time_taken.append(PPO.train())

    print(time_taken)