#!/home/alex/Desktop/Exjobb/RL_collision_avoidance/RL_collision_avoidance_env/bin/python

import sys
import os
import numpy as np

import pybullet as p
import pkg_resources
from gymnasium import spaces
import pybullet_data
import time

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from ament_index_python.packages import get_package_share_directory

def main():
    print('Hi from simulator.')
    #physicsClient = p.connect(p.GUI)

    env = BaseRLAviary(gui=True)

    #p.setAdditionalSearchPath(pybullet_data.getDataPath())

    cf2p_urdf_path = '/home/alex/Desktop/Exjobb/RL_collision_avoidance/gym-pybullet-drones/gym_pybullet_drones/assets/cf2p.urdf'

    
    #p.setGravity(0, 0, -9.81)
    #planeID = p.loadURDF("plane.urdf")
    #startPos = [0, 0, 1]
    #startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    #boxId = p.loadURDF(cf2p_urdf_path, startPos, startOrientation)
    #p.resetBasePositionAndOrientation(boxId, startPos, startOrientation)

    action = np.array([[-1, 0]])
    for i in range (10000):
        #p.stepSimulation()
        env.step(action)
        time.sleep(1./240.)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    print("Cube position:", cubePos)
    #p.disconnect()


if __name__ == '__main__':
    main()
