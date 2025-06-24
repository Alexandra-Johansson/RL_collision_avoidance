import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel
from train_PPO import Train_PPO

parameters = {
    'drone_model': DroneModel.CF2X,
    'num_objects': 1,  # Number of balls to add
    'initial_xyzs': np.array([[1.0, 2.0, 1.0]]),
    'ctrl_freq': 60,
    'pyb_freq': 240,
    'action_size': 0.2,
    'eval_freq': 10,  # Evaluate every 1 episodes
    'target_pos': np.array([[.0, .0, 1.0]]),
    'target_radius': 0.1,  # Radius of the target sphere
    'avoidance_radius': 0.75,  # Radius of the avoidance sphere
    'episode_length': 15,  # seconds
    'eval_episodes' : 10,  # Number of episodes to evaluate
    'learning_rate': 1e-3,
    'batch_size': 64,
    'num_epochs': 20,
    'train_freq': 1,  # Train every step
    'gradient_steps': -1,  # Number of gradient steps per update
    'obs_noise': False,  # Add noise to the observations
    'obs_noise_std': 0.01,  # Standard deviation of the noise
    'reward_collision': -500.0,
    'reward_terminated': 1000.0,
    'reward_target_distance': -10.0,
    'reward_target_distance_delta': 200.0, # Positive for rewarding moving towards target
    'reward_angular_velocity_delta': -5.0, # Negative reward for changing angle
    'reward_object_distance_delta': 10.0, # Positive for rewarding moving away from object
    'reward_step': -50,
    'reward_in_target': 100.0,
    'target_reward': 150000.0,  # Reward to stop training
    'total_timesteps': int(1.1*1e6),  # Total timesteps to train
    'nr_of_env': 10,  # Number of environments to train in parallel
}

if __name__ == "__main__":

    PPO = Train_PPO(parameters=parameters, train_gui=False)

    PPO.train()