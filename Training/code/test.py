import numpy as np
import time
import math

from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType

from RLEnv import RLEnv

parameters = {
    'drone_model': DroneModel.CF2X,
    'initial_xyzs': np.array([[1, 0, 1]]),
    'ctrl_freq': 240,
}

env = RLEnv(parameters=parameters,gui=True)

action = np.array([[0.0, 0.0, 1.0]])

for i in range (10000):
        '''
        x = math.cos(4*2*math.pi*i/10000)
        y = math.sin(4*2*math.pi*i/10000)
        action[0,0] = x
        action[0,1] = y
        '''
        
        if (i%100 == 0):
            # Add a ball at a random position
            #env.addBallRandom()

            # Add a ball at a fixed position for collision testing
            #env.addBall([0, 0, 0.5], [0, 0, 100])
        

        env.step(action)
        time.sleep(1./240.)