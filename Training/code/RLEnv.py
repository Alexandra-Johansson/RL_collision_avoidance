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
        self.NUM_OBJECTS = parameters['num_objects']
        self.INITIAL_XYZS = parameters['initial_xyzs']
        self.CTRL_FREQ = parameters['ctrl_freq']
        self.TARGET_POS = parameters['target_pos']
        self.ACT4 = parameters['act4']
        self.OBS_NOISE = parameters['obs_noise']
        self.OBS_NOISE_STD = parameters['obs_noise_std']

        self.ORIGINAL_XYZS = np.copy(self.INITIAL_XYZS)

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

        self.EPISODE_LEN_SEC = parameters['episode_length']

        drone_id = 0
        drone_state_vec = self._getDroneStateVector(drone_id)

        self.reward_state = drone_state_vec[0:3]  # Position
        self.target_distance = np.linalg.norm(self.TARGET_POS[0] - self.reward_state)
        self.ang_vel = drone_state_vec[13:16]  # Angular velocity

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Compute the reward, termination, truncation, and info for one step

        if self._getCollision(self.DRONE_IDS[0]):
            reward = self._computeReward()

            self.reset_drone()
            
        return obs, reward, terminated, truncated, info

    def reset(self, seed = None, options = None):
        drone_id = 0
        self.ctrl[drone_id].reset()

        obs, info = super().reset(seed, options)

        drone_state_vec = self._getDroneStateVector(drone_id)
        self.reward_state = drone_state_vec[0:3]  # Position
        self.target_distance = np.linalg.norm(self.TARGET_POS[0] - drone_state_vec[0:3])
        self.ang_vel = drone_state_vec[13:16] # Angular velocity

        return obs, info

    def _computeReward(self):
        # TODO
        ret = 0

        return ret

    def _computeTerminated(self):
        Terminated = False

        drone_id = 0
        drone_state_vec = self._getDroneStateVector(drone_id)
        
        # If the drone has avoided all obstacles, terminate the episode
        i = 0
        for ball in self.ball_list:
            i += 1
            vel, ang_v = p.getBaseVelocity(ball, physicsClientId = self.CLIENT)
            pos, quat = p.getBasePositionAndOrientation(ball, physicsClientId = self.CLIENT)
            if (vel[2] > 1e-4) or (pos[2] > 1e-1):
                # If a ball is in the air don't terminate
                return Terminated
        
        # If no ball is in the air terminate
        Terminated = True
        return Terminated

    def _computeTruncated(self):
        Truncated = False

        drone_id = 0
        drone_state_vec = self._getDroneStateVector(drone_id)

        if (self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC):
            Truncated = True
            print("Time limit reached, episode truncated.")
        
        elif (abs(drone_state_vec[0]) > 5 or
            abs(drone_state_vec[1]) > 5 or
            abs(drone_state_vec[2]) > 5):
            Truncated = True
            print("Drone out of bounds, episode truncated.")

        elif (abs(drone_state_vec[7]) > .9 or
                abs(drone_state_vec[8]) > .9 or
                abs(drone_state_vec[9]) > .9):
            Truncated = True
            print("Drone orientation out of bounds, episode truncated.")

        return Truncated

    def _getCollision(self, obj):

        constact_points = p.getContactPoints(obj, physicsClientId=self.CLIENT)

        if len(constact_points) > 0:
            return True
        else:
            return False

    def _obesvationSpace(self):
        if self.OBS_TYPE == ObservationType and self.ACT_TYPE == ActionType.PID:
            pos_low = np.array([-5, -5, 0])
            pos_high = np.array([5, 5, 5])

            # Add drone obesrvation space
            obs_drone_lower_bound = np.array([[pos_low] for drone in range(self.NUM_DRONES)])
            obs_drone_upper_bound = np.array([[pos_high] for drone in range(self.NUM_DRONES)])

            # Add object observation space
            obs_obj_lower_bound = np.hstack([obs_lower_bound, np.array([[pos_low] for obj in range(self.NUM_OBJECTS)])])
            obs_obj_upper_bound = np.hstack([obs_upper_bound, np.array([[pos_high] for obj in range(self.NUM_OBJECTS)])])

            return spaces.Dict({
                "Drone_position": spaces.Box(low=obs_drone_lower_bound, high=obs_drone_upper_bound, dtype=np.float32),
                "Object_position": spaces.Box(low=obs_obj_lower_bound, high=obs_obj_upper_bound, dtype=np.float32)})
        else:
            super()._obesvationSpace()

    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.KIN and self.ACT_TYPE == ActionType.PID:
            if self.OBS_NOISE:
                #noise_drone = np.random.normal(0, self.OBS_NOISE_STD, (self.NUM_DRONES, 3))
                noise_obj = np.random.normal(0, self.OBS_NOISE_STD, (self.NUM_OBJECTS, 3))
            else:
                #noise_drone = np.zeros((self.NUM_DRONES, 3))
                noise_obj = np.zeros((self.NUM_OBJECTS, 3))

            
            drone_pos = np.zeros((self.NUM_DRONES, 3))
            for i in range(self.NUM_DRONES):
                obs = self._getDroneStateVector(i)
                drone_pos[i,:] = obs[0:3]# + noise_drone[drone,:]

            obj_pos = np.zeros((self.NUM_OBJECTS,3))
            for i in range(self.NUM_OBJECTS):
                obs, _ = p.getBasePositionAndOrientation(self.ball_list[i], physicsClientId=self.CLIENT)
                obj_pos[i,:] = obs[0:3] + noise_obj[i,:]

            return {
                "Drone_position": drone_pos,
                "Object_position": obj_pos
            }


        else:
            return super()._computeObs()

    def _computeInfo(self):
        return {"info": 0}

    def reset_drone(self):
        drone_id = 0

        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.quat[0,3] = 1
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_vel = np.zeros((self.NUM_DRONES, 3))

        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[drone_id],
            self.INITIAL_XYZS[drone_id,:],
            p.getQuaternionFromEuler(self.INIT_RPYS[drone_id,:]),
            physicsClientId=self.CLIENT)
        p.resetBaseVelocity(
            self.DRONE_IDS[drone_id],
            self.vel[drone_id,:],
            self.ang_vel[drone_id,:],
            physicsClientId=self.CLIENT)
        # Reset the drone state vector

        self.ctrl[drone_id].reset

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
    
    def getAction(self):
        # TODO
        super().getAction()
        pass