import os
import numpy as np
import math

import pybullet as p
from gymnasium import spaces
import pybullet_data

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class RLEnv(BaseRLAviary):
    def __init__(self,
                 num_drones = 1,
                 neighborhood_radius = np.inf,
                 initial_rpys = None,
                 physics = Physics.PYB,
                 pyb_freq = 240,
                 gui = False,
                 record = False,
                 obs = ObservationType.KIN,
                 act = ActionType.PID,
                 parameters = None):

        self.DRONE_MODEL = parameters['drone_model']
        self.INITIAL_XYZS = parameters['initial_xyzs']
        self.CTRL_FREQ = parameters['ctrl_freq']

        self.FORCE_MAG_MIN_XY = 75
        self.FORCE_MAG_MAX_XY = 100
        self.FORCE_MAG_MIN_Z = 50
        self.FORCE_MAG_MAX_Z = 100
        self.FORCE_VAR = 5

        self.ball_list = []

        super().__init__(self.DRONE_MODEL,
                         num_drones,
                         neighborhood_radius,
                         self.INITIAL_XYZS,
                         initial_rpys,
                         physics,
                         pyb_freq,
                         self.CTRL_FREQ,
                         gui,
                         record,
                         obs,
                         act)

    def _computeReward(self):
        ret = 0

        return ret

    def _computeTerminated(self):
        Terminated = False

        return Terminated

    def _computeTruncated(self):
        Truncated = False

        return Truncated

    def _computeInfo(self):
        info = {}

        return info

    def _addObstacles(self):
        # Add initial obstacles or environment features here if needed
        
        pass

    def addBall(self,position=[0.0,0.0,0.0],force=[0.0,0.0,0.0]):
        # position: where the ball will be added
        # force: the force applied to the ball at the moment of creation

        search_path = "/home/alex/Desktop/Exjobb/RL_collision_avoidance/Training/resources"
        p.setAdditionalSearchPath(search_path)
        
        self.ball_list.append(p.loadURDF("custom_sphere_small.urdf",
                       basePosition=(position[0], position[1], position[2])))
        
        position = [0, 0, 0]  # Relative position (center of mass)
        p.applyExternalForce(
            objectUniqueId=self.ball_list[-1],  # Get the last added ball
            linkIndex=-1,  # -1 means base/root link
            forceObj=force,
            posObj=position,
            flags=p.WORLD_FRAME
        )
    
    def addBallRandom(self):
        # Randomly generate position and force for the ball

        # Generate a random position for the ball
        angle = np.random.uniform(-math.pi, math.pi)
        magnitude = np.random.uniform(2, 3)
        x_ball = magnitude * math.cos(angle)
        y_ball = magnitude * math.sin(angle)
        z_ball = np.random.uniform(0.75, 1.25)

        # Calculate force based on the position
        norm = np.linalg.norm([x_ball, y_ball])
        force_x = -np.random.uniform(self.FORCE_MAG_MIN_XY,self.FORCE_MAG_MAX_XY) * x_ball/norm +  \
                                                            np.random.uniform(-self.FORCE_VAR,self.FORCE_VAR)
        force_y = -np.random.uniform(self.FORCE_MAG_MIN_XY,self.FORCE_MAG_MAX_XY) * y_ball/norm + \
                                                            np.random.uniform(-self.FORCE_VAR,self.FORCE_VAR)
        force_z = np.random.uniform(self.FORCE_MAG_MIN_Z,self.FORCE_MAG_MAX_Z) + \
                                                            np.random.uniform(-self.FORCE_VAR,self.FORCE_VAR)

        # Add the ball with the generated position and force
        self.addBall(position=[x_ball, y_ball, z_ball], force=[force_x, force_y, force_z])

        pass