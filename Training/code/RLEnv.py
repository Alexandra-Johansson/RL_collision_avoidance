import os
import numpy as np

import pybullet as p
from gymnasium import spaces

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

        self.drone_model = parameters['drone_model']
        self.initial_xyzs = parameters['initial_xyzs']
        self.CTRL_FREQ = parameters['ctrl_freq']

        print("RLEnv: drone_model:", self.drone_model)

        super().__init__(self.drone_model,
                         num_drones,
                         neighborhood_radius,
                         self.initial_xyzs,
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