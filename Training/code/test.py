import numpy as np
import time
import math

from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType

from RLEnv import RLEnv

parameters = {
    'drone_model': DroneModel.CF2X,
    'num_objects': 1,  # Number of balls to add
    'initial_xyzs': np.array([[1, 2, 0.1]]),
    'ctrl_freq': 240,
    'target_pos': np.array([[.0, .0, 1.0]]),
    'target_radius': 0.25,  # Radius of the target sphere
    'avoidance_radius': 0.75,  # Radius of the avoidance sphere
    'episode_length': 60,  # seconds
    'act4': True, # Use actions x,y,z and velocity or only position
    'obs_noise': False,  # Add noise to the observations
    'obs_noise_std': 0.01,  # Standard deviation of the noise
    'reward_collision': 200.0,
    'reward_terminated': 100.0,
    'reward_target_distance_delta': 10.0,
    'reward_object_distance_delta': 10.0,
    'reward_step': 0.01,
    'reward_in_target': 1.0
}

env = RLEnv(parameters=parameters,gui=True)

action = np.array([[.0, .0, 1.0]])

for i in range (10000):
        '''
        x = math.cos(4*2*math.pi*i/10000)
        y = math.sin(4*2*math.pi*i/10000)
        action[0,0] = x
        action[0,1] = y
        '''
        if (i%1000 == 0):
            # Add a ball at a random position
            env.addBallRandom()

            # Add a ball at a fixed position for collision testing
            #env.addBall([0, 0, 0.5], [0, 0, 100])

            #env.reset()
            pass
        

        obs, reward, terminated, truncated, info = env.step(action)
        #print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated, "\tInfo:", info)
        time.sleep(1./240.)