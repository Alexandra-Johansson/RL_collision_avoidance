import math
import os
import time

import numpy as np
import pybullet as pb
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    ActionType,
    ObservationType,
    Physics,
)
from gymnasium import spaces
from predictor import KalmanFilter
from scipy.spatial.transform import Rotation


class RLEnv(BaseRLAviary):
    def __init__(self,
                 num_drones = 1,
                 neighborhood_radius = np.inf,
                 initial_rpys = None,
                 physics = Physics.PYB,
                 obs = ObservationType.KIN,
                 #act = ActionType.PID,
                 parameters = None):

        self.DRONE_MODEL = parameters['drone_model']
        self.ACTION_TYPE = parameters['action_type']
        self.NUM_OBJECTS = parameters['num_objects']
        self.INITIAL_XYZS = parameters['initial_xyzs']
        self.CTRL_FREQ = parameters['ctrl_freq']
        self.PYB_FREQ = parameters['pyb_freq']
        self.EPISODE_LEN = parameters['episode_length']
        self.ACTION_SIZE = parameters['action_size']
        self.VELOCITY_SIZE = parameters['velocity_size']
        self.GOAL_POS = parameters['goal_pos']
        self.GOAL_RADIUS = parameters['goal_radius']
        self.AVOIDANCE_RADIUS = parameters['avoidance_radius']
        self.CRITICAL_SAFETY_DISTANCE = parameters['critical_safety_distance']
        self.OBS_NOISE = parameters['obs_noise']
        self.OBS_NOISE_STD = parameters['obs_noise_std']
        self.PROCESS_NOISE = parameters['kf_process_noise']
        self.MEASUREMENT_NOISE = parameters['kf_measurement_noise']
        self.SRT = parameters['space_reduction_transformation']
        self.GUI = parameters['gui']
        self.RECORD = parameters['record']

        self.REWARD_COLLISION = parameters['reward_collision']
        self.REWARD_SUCESS = parameters['reward_sucess']
        self.REWARD_END_OUTSIDE_GOAL = parameters['reward_end_outside_goal']
        self.REWARD_TRUNCATION = parameters['reward_truncation']
        self.REWARD_GOAL_DISTANCE = parameters['reward_goal_distance']
        self.REWARD_GOAL_DISTANCE_DELTA = parameters['reward_goal_distance_delta']
        self.REWARD_RPY = parameters['reward_rpy']
        self.REWARD_ANGULAR_VELOCITY_DELTA = parameters['reward_angular_velocity_delta']
        self.REWARD_OBJECT_DISTANCE = parameters['reward_object_distance']
        self.REWARD_OBJECT_DISTANCE_DELTA = parameters['reward_object_distance_delta']
        self.REWARD_STEP = parameters['reward_step']
        self.REWARD_ACTION_DIFFERENCE = parameters['reward_action_difference']

        self.OBS_TIMESTEP = parameters['obs_timestep']
        self.OBS_OBJ_VEL = parameters['obs_obj_vel']
        self.OBS_KF = parameters["obs_kf"]
        self.OBS_ACTION_DIFFERENCE = parameters['obs_action_difference']

        self.ORIGINAL_XYZS = np.copy(self.INITIAL_XYZS)

        self.VELOCITY_VAR = 0
        self.DRONE_ID = 0

        self.ball_list = []
        self.time_limit_reached = False
        self.truncation_reason = True
        self.collision_type = True
        self.rot_angle = 0.0
        
        super().__init__(self.DRONE_MODEL,
                         num_drones,
                         neighborhood_radius,
                         self.INITIAL_XYZS,
                         initial_rpys,
                         physics,
                         self.PYB_FREQ,
                         self.CTRL_FREQ,
                         self.GUI,
                         self.RECORD,
                         obs,
                         self.ACTION_TYPE
                         )

        self.EPISODE_LEN_SEC = parameters['episode_length']

        # Add more filters if multiple objects should be tracked
        # TODO, needs work before multiple objects are supported
        self.kf = KalmanFilter(dt = self.CTRL_TIMESTEP, 
                               process_var = self.PROCESS_NOISE,
                               measurement_var = self.MEASUREMENT_NOISE,
                               gravity = self.G)
        
        self.kf_pos_error = np.zeros((self.NUM_OBJECTS, 3))
        self.kf_vel_error = np.zeros((self.NUM_OBJECTS, 3))

        self.REWARD_IN_GOAL_CONSTANT = self.REWARD_GOAL_DISTANCE/(self.GOAL_RADIUS**2) + self.GOAL_RADIUS**2
        
        drone_state_vec = self._getDroneStateVector(self.DRONE_ID)

        self.curr_drone_pos = drone_state_vec[0:3]  # Position
        self.curr_drone_rpy = drone_state_vec[7:10]  # Roll, Pitch, Yaw
        self.goal_distance = np.linalg.norm(self.GOAL_POS[0] - drone_state_vec[0:3])
        self.curr_ang_vel = drone_state_vec[13:16]  # Angular velocity
        self.obj_distances = np.zeros((self.NUM_OBJECTS, 1))
        if self.SRT:
            self.prev_obj_pos = np.zeros((self.NUM_OBJECTS, 2))  # Previous object positions
        else:
            self.prev_obj_pos = np.zeros((self.NUM_OBJECTS, 3))  # Previous object positions
        
        if self.ACTION_TYPE == ActionType.PID:
            self.curr_action = np.zeros((1, 3))  # Current action
            self.prev_action = np.zeros((1, 3))  # Previous action
        elif self.ACTION_TYPE == ActionType.VEL:
            self.curr_action = np.zeros((1, 4))
            self.prev_action = np.zeros((1, 4))

        self.visual_goal_id = pb.createVisualShape(shapeType = pb.GEOM_SPHERE,
                             rgbaColor=[0.0, 1.0, 0.0, 0.5], 
                             radius=self.GOAL_RADIUS, 
                             physicsClientId=self.CLIENT)
        
        pb.createMultiBody(baseMass=0,
                           baseVisualShapeIndex=self.visual_goal_id,
                           basePosition=self.GOAL_POS[0],
                           physicsClientId=self.CLIENT)
        
        self.visual_action_ids = []

        self.reward_rpy = 0.0
        self.reward_goal_distance = 0.0
        self.reward_object_distance = 0.0
        self.reward_object_distance_delta = 0.0
        self.reward_action_difference = 0.0

        self.min_obj_distance = np.inf
        self.max_goal_distance = 0.0
        self.max_kf_pos_error = 0.0
        self.average_kf_vel_error = 0.0
        self.average_kf_vel_error_counter = 1

    def step(self, rel_action):
        # Convert the scaled relative action to a global action
        if self.ACTION_TYPE == ActionType.PID:
            rot_action = self.reverse_rotate_vector_from_x_axis(rel_action)
            rel_action = np.array([[rot_action[0], rot_action[1], rot_action[2]]])
            action = self.curr_drone_pos + rel_action*self.ACTION_SIZE
        if self.ACTION_TYPE == ActionType.VEL:
            pos_and_vel = np.append(self.curr_drone_pos, 0)
            #TODO: FIX ROTATION OF ACTION
            rel_action[0][0:3] = self.reverse_rotate_vector_from_x_axis(rel_action[0][0:3])
            rel_action[0][0:3] = rel_action[0][0:3]*self.ACTION_SIZE
            rel_action[0][3] = rel_action[0][3]*self.VELOCITY_SIZE
            action = pos_and_vel + rel_action
        #action = rel_action
        self.curr_action = rel_action
        '''
        if self.GUI:
            self.visual_action_id = pb.createVisualShape(shapeType = pb.GEOM_SPHERE,
                                rgbaColor=[1.0, 0.0, 0.0, 1.0], 
                                radius=0.01, 
                                physicsClientId=self.CLIENT)
            
            self.visual_action_ids.append(pb.createMultiBody(baseMass=0,
                            baseVisualShapeIndex=self.visual_action_id,
                            basePosition=action[0][0:3],
                            physicsClientId=self.CLIENT))
            
            for id in self.visual_action_ids:
                data = pb.getVisualShapeData(id, physicsClientId=self.CLIENT)
                rgba = np.array(data[0][7])  # Get the rgba color of the visual shape
                rgba[3] = rgba[3] - 0.1
                pb.changeVisualShape(objectUniqueId = id,
                                    linkIndex = data[0][1],
                                    rgbaColor=rgba,            #pb.removeBody(self.visual_action_ids.pop(0), physicsClientId=self.CLIENT)
                                    physicsClientId=self.CLIENT)
            if len(self.visual_action_ids) >= 10:
                self.visual_action_ids.pop(0)
        '''
        self.kf.timeUpdate()

        # Simulate a step and compute the 
        # observation, reward, termination, truncation, and info
        obs, reward, terminated, truncated, info = super().step(action)

        # Compute info for logging
        self.min_obj_distance = min(self.min_obj_distance, np.linalg.norm(obs["Object_position"]))
        self.max_goal_distance = max(self.max_goal_distance, np.linalg.norm(obs["Goal_position"]))

        # If GUI is enabled, sleep to visualize the simulation properly
        if self.GUI:
            time.sleep(self.CTRL_TIMESTEP)
            
        return obs, reward, terminated, truncated, info

    def reset(self, seed = None, options = None):
        self.ctrl[self.DRONE_ID].reset()
        
        for ball in self.ball_list:
            try:
                pb.removeBody(ball, physicsClientId=self.CLIENT)
                self.ball_list.pop(0)
            except Exception:
                # If the ball is already removed, ignore the error
                pass

        obs, info = super().reset(seed, options)

        self.reset_drone()

        drone_state_vec = self._getDroneStateVector(self.DRONE_ID)
        self.curr_drone_pos = drone_state_vec[0:3]  # Position
        self.curr_drone_rpy = drone_state_vec[7:10]  # Roll, Pitch, Yaw
        self.goal_distance = np.linalg.norm(self.GOAL_POS[0] - self.curr_drone_pos)
        self.curr_ang_vel = drone_state_vec[13:16] # Angular velocity
        self.obj_distances = np.zeros((self.NUM_OBJECTS, 1))
        if self.SRT:
            self.prev_obj_pos = np.zeros((self.NUM_OBJECTS, 2))  # Previous object positions
        else:
            self.prev_obj_pos = np.zeros((self.NUM_OBJECTS, 3))  # Previous object positions
        if self.ACTION_TYPE == ActionType.PID:
            self.act_size = 3
        elif self.ACTION_TYPE == ActionType.VEL:
            self.act_size = 4
        self.curr_action = np.zeros((1, self.act_size))  # Current action
        self.prev_action = np.zeros((1, self.act_size))  # Previous action for action difference observation
        self.time_limit_reached = False

        self.min_obj_distance = np.inf
        self.max_goal_distance = 0.0
        self.max_kf_pos_error = 0.0
        self.average_kf_vel_error = 0.0
        self.average_kf_vel_error_counter = 1

        self.kf_pos_error = np.zeros((self.NUM_OBJECTS, 3))
        self.kf_vel_error = np.zeros((self.NUM_OBJECTS, 3))
        self.visual_action_ids = []

        self.visual_goal_id = pb.createVisualShape(shapeType = pb.GEOM_SPHERE,
                             rgbaColor=[0.0, 1.0, 0.0, 0.5], 
                             radius=self.GOAL_RADIUS, 
                             physicsClientId=self.CLIENT)

        pb.createMultiBody(baseMass=0,
                           baseVisualShapeIndex=self.visual_goal_id,
                           basePosition=self.GOAL_POS[0],
                           physicsClientId=self.CLIENT)

        self.addBallRandom()

        return obs, info

    def _computeReward(self):
        ret = 0.0

        self.reward_rpy = 0.0
        self.reward_goal_distance = 0.0
        self.reward_object_distance = 0.0
        self.reward_object_distance_delta = 0.0 #TODO: Use estimated velocity from kf instead
        self.reward_action_difference = 0.0

        prev_ang_vel = np.copy(self.curr_ang_vel)
        prev_goal_distance = np.copy(self.goal_distance)
        prev_obj_distances = np.copy(self.obj_distances)

        drone_state_vec = self._getDroneStateVector(self.DRONE_ID)

        self.curr_drone_pos = drone_state_vec[0:3]
        self.curr_drone_rpy = drone_state_vec[7:10]
        self.curr_ang_vel = drone_state_vec[13:16]
        
        self.goal_distance = np.linalg.norm(self.GOAL_POS[0] - self.curr_drone_pos)

        self.obj_distances = np.zeros((len(self.ball_list), 1))
        # Calculate the distance to each object
        for i in range(len(self.ball_list)):
            ball_pos, _ = pb.getBasePositionAndOrientation(self.ball_list[i], physicsClientId=self.CLIENT)
            self.obj_distances[i] = np.linalg.norm(ball_pos[0:3] - self.curr_drone_pos)

        
        # Check if the episode is terminated
        if self._computeTerminated():
            ret = self.REWARD_COLLISION
            # Negative reward for collision
        elif self._computeTruncated():
            if self.time_limit_reached:
                if (self.goal_distance <= self.GOAL_RADIUS):
                    ret = self.REWARD_SUCESS
                else:
                    ret = self.REWARD_END_OUTSIDE_GOAL
            else:
                # Negative reward for truncation
                ret = self.REWARD_TRUNCATION
        # Else calculate the reward based on the states
        
        if False:
            pass
        else:
            obj_distances_delta = np.sum(prev_obj_distances - self.obj_distances)
            
            if (self.goal_distance <= self.GOAL_RADIUS):
                self.reward_goal_distance = -self.goal_distance**2 + self.REWARD_IN_GOAL_CONSTANT
            else:
                self.reward_goal_distance = self.REWARD_GOAL_DISTANCE / (self.goal_distance**2)
                # TODO, add goal_distance_delta
            
            # TODO, needs work before multiple objects are supported, change to loop trough all objects
            #if (any(self.obj_distances < self.AVOIDANCE_RADIUS)):
            self.reward_object_distance = self.REWARD_OBJECT_DISTANCE / np.sum(self.obj_distances)**2

            if self.OBS_ACTION_DIFFERENCE:
                self.reward_action_difference = self.REWARD_ACTION_DIFFERENCE*np.linalg.norm(self.action_difference)

        #ret += self.reward_rpy
        ret += self.reward_goal_distance
        ret += self.reward_object_distance
        #ret += self.reward_object_distance_delta
        ret += self.reward_action_difference

        return float(ret)

    def _computeTerminated(self):
        Terminated = False
        
        if self._getCollision(self.DRONE_IDS[0]):
            Terminated = True

        return Terminated

    def _computeTruncated(self):
        Truncated = False
        self.truncation_reason = None
        self.time_limit_reached = False

        drone_state_vec = self._getDroneStateVector(self.DRONE_ID)

        # Check if the episode length is reached
        if (self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC):
            Truncated = True
            self.time_limit_reached = True
            self.truncation_reason = "time_limit"
        # Check if the drone is out of bounds
        elif (abs(drone_state_vec[0]) > 5 or
            abs(drone_state_vec[1]) > 5 or
            abs(drone_state_vec[2]) > 5):
            Truncated = True
            self.truncation_reason = "out_of_bounds"
        # Check if the drone orientation is out of bounds
        elif (abs(drone_state_vec[7]) > .9 or
                abs(drone_state_vec[8]) > .9):
            Truncated = True
            self.truncation_reason = "orientation"

        return Truncated

    def _getCollision(self, obj):
        self.collision_type = None
        constact_points = pb.getContactPoints(obj, physicsClientId=self.CLIENT)

        if any(self.obj_distances < self.CRITICAL_SAFETY_DISTANCE):
            self.collision_type = "object_collision"
            return True
        elif len(constact_points) > 0:
            self.collision_type = "contact_collision"
            return True
        else:
            return False

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.KIN and (self.ACT_TYPE == ActionType.PID or self.ACT_TYPE == ActionType.VEL):
            vel_low = np.array(-5*np.ones(3))
            vel_high = np.array(5*np.ones(3))
            if self.SRT:
                rpy_low = np.array(-1*np.ones(6))
                rpy_high = np.array(1*np.ones(6))
            else:
                rpy_low = np.array(-1*np.ones(3))
                rpy_high = np.array(1*np.ones(3))
            rpy_vel_low = np.array([-10, -10, -6])
            rpy_vel_high = np.array([10, 10, 6])
            altitude_low = np.array([-0.5])
            altitude_high = np.array([10.0])

            if self.SRT:
                obj_pos_low = np.array(-5*np.ones(2))
                obj_pos_high = np.array(5*np.ones(2))
            else:
                obj_pos_low = np.array(-5*np.ones(3))
                obj_pos_high = np.array(5*np.ones(3))
            obj_vel_low = np.array(-20*np.ones(3))
            obj_vel_high = np.array(20*np.ones(3))
            
            goal_pos_low = np.array(-5*np.ones(3))
            goal_pos_high = np.array(5*np.ones(3))

            timestep_low = np.array([0])
            timestep_high = np.array([self.EPISODE_LEN*self.PYB_FREQ + 1])

            if self.ACTION_TYPE == ActionType.PID:
                action_difference_low = np.array(-1*np.ones(3))
                action_difference_high = np.array(1*np.ones(3))
            elif self.ACTION_TYPE == ActionType.VEL:
                action_difference_low = np.array([-1.0, -1.0, -1.0, -5.0])
                action_difference_high = np.array([1.0, 1.0, 1.0, 5.0])

            # Add drone observation space
            obs_drone_vel_lower_bound = np.array([vel_low])
            obs_drone_vel_upper_bound = np.array([vel_high])
            obs_drone_rpy_lower_bound = np.array([rpy_low])
            obs_drone_rpy_upper_bound = np.array([rpy_high])
            obs_drone_rpy_vel_lower_bound = np.array([rpy_vel_low])
            obs_drone_rpy_vel_upper_bound = np.array([rpy_vel_high])
            obs_altitude_lower_bound = np.array([altitude_low])
            obs_altitude_upper_bound = np.array([altitude_high])

            # Add goal observation space
            obs_goal_pos_lower_bound = np.array([goal_pos_low])
            obs_goal_pos_upper_bound = np.array([goal_pos_high])

            # Add timestep observation space
            timestep_lower_bound = np.array([timestep_low])
            timestep_upper_bound = np.array([timestep_high])

            # Add object observation space
            obs_obj_lower_bound = np.array([obj_pos_low for obj in range(self.NUM_OBJECTS)])
            obs_obj_upper_bound = np.array([obj_pos_high for obj in range(self.NUM_OBJECTS)])
            obs_obj_vel_lower_bound = np.array([obj_vel_low for obj in range(self.NUM_OBJECTS)])
            obs_obj_vel_upper_bound = np.array([obj_vel_high for obj in range(self.NUM_OBJECTS)])

            # Add action difference observation space
            obs_action_difference_lower_bound = np.array([action_difference_low])
            obs_action_difference_upper_bound = np.array([action_difference_high])

            obs_dict = {
                    "Drone_velocity": spaces.Box(low=obs_drone_vel_lower_bound.flatten(), high=obs_drone_vel_upper_bound.flatten(), dtype=np.float64),
                    "Drone_rpy": spaces.Box(low=obs_drone_rpy_lower_bound.flatten(), high=obs_drone_rpy_upper_bound.flatten(), dtype=np.float64),
                    "Drone_rpy_velocity": spaces.Box(low=obs_drone_rpy_vel_lower_bound.flatten(), high=obs_drone_rpy_vel_upper_bound.flatten(), dtype=np.float64),
                    "Drone_altitude": spaces.Box(low=obs_altitude_lower_bound.flatten(), high=obs_altitude_upper_bound.flatten(), dtype=np.float64),
                    "Goal_position": spaces.Box(low=obs_goal_pos_lower_bound.flatten(), high=obs_goal_pos_upper_bound.flatten(), dtype=np.float64),
                    "Object_position": spaces.Box(low=obs_obj_lower_bound.flatten(), high=obs_obj_upper_bound.flatten(), dtype=np.float64),
                    "Object_position_t-1": spaces.Box(low=obs_obj_lower_bound.flatten(), high=obs_obj_upper_bound.flatten(), dtype=np.float64)}

            if self.OBS_TIMESTEP:
                obs_dict["Timestep"] = spaces.Box(low=timestep_lower_bound.flatten(), high=timestep_upper_bound.flatten(), dtype=np.float64)

            if self.OBS_OBJ_VEL:
                obs_dict["Object_velocity"] = spaces.Box(low=obs_obj_vel_lower_bound.flatten(), high=obs_obj_vel_upper_bound.flatten(), dtype=np.float64)
            
            if self.OBS_ACTION_DIFFERENCE:
                obs_dict["Action_difference"] = spaces.Box(low=obs_action_difference_lower_bound.flatten(), high=obs_action_difference_upper_bound.flatten(), dtype=np.float64)

            return spaces.Dict(obs_dict)

        else:
            super()._observationSpace()
    
    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.KIN and (self.ACT_TYPE == ActionType.PID or self.ACT_TYPE == ActionType.VEL):
            if self.OBS_NOISE:
                #noise_drone = np.random.normal(0, self.OBS_NOISE_STD, (1, 3))
                noise_obj = np.random.normal(0, self.OBS_NOISE_STD, (self.NUM_OBJECTS, 3))
            else:
                #noise_drone = np.zeros((1, 3))
                noise_obj = np.zeros((self.NUM_OBJECTS, 3))

            # Compute drone observations
            drone_pos = np.zeros((1, 3))
            drone_vel = np.zeros((1, 3))
            drone_rpy = np.zeros((1, 3))
            drone_rpy_vel = np.zeros((1, 3))
            drone_altitude = np.zeros((1, 1))

            obs = self._getDroneStateVector(self.DRONE_ID)
            drone_pos[0,:] = obs[0:3]
            drone_vel[0,:] = obs[10:13]
            drone_rpy[0,:] = obs[7:10]
            drone_rpy_vel[0,:] = obs[13:16]
            drone_altitude[0, :] = obs[2]

            # Compute object observations
            obj_pos = np.zeros((self.NUM_OBJECTS, 3))
            if self.SRT:
                obj_pos_return = np.zeros((self.NUM_OBJECTS, 2))
            else:
                obj_pos_return = np.zeros((self.NUM_OBJECTS, 3))
            obj_vel = np.zeros((self.NUM_OBJECTS, 3))
            obj_pos_pb = np.zeros((self.NUM_OBJECTS, 3))
            obj_vel_pb = np.zeros((self.NUM_OBJECTS, 3))
            obj_pos_kf = np.zeros((self.NUM_OBJECTS, 3))
            obj_vel_kf = np.zeros((self.NUM_OBJECTS, 3))
            # TODO: FIX FOR MULTIPLE OBJECTS AND FIX NOISE
            for i in range(self.NUM_OBJECTS):
                if i < len(self.ball_list):
                    obs_pos_pb, _ = pb.getBasePositionAndOrientation(self.ball_list[i], physicsClientId=self.CLIENT)
                    obs_vel_pb, _ = pb.getBaseVelocity(self.ball_list[i], physicsClientId=self.CLIENT)
                    self.kf.measurementUpdate(obs_pos_pb[0:3] + noise_obj[i, :])
                    obs_kf_state = self.kf.getState()

                    obj_pos_kf[i, :] = obs_kf_state[0:3, 0]
                    obj_pos_pb[i, :] = obs_pos_pb[0:3]
                    self.kf_pos_error[i, :] = obj_pos_pb[i, :] - obj_pos_kf[i, :]
                    # TODO, Fix so multiple objects are supported
                    self.max_kf_pos_error = max(self.max_kf_pos_error, np.linalg.norm(self.kf_pos_error[i, :]))

                    obj_vel_kf[i, :] = obs_kf_state[3:6, 0]
                    obj_vel_pb[i, :] = obs_vel_pb[0:3]
                    self.kf_vel_error[i, :] = obj_vel_pb[i, :] - obj_vel_kf[i, :]
                    # TODO, Fix so multiple objects are supported
                    self.average_kf_vel_error = ((self.average_kf_vel_error_counter-1)*self.average_kf_vel_error + np.linalg.norm(self.kf_vel_error[i, :]))/self.average_kf_vel_error_counter
                    self.average_kf_vel_error_counter += 1

                    obj_pos_pb[i, :] = obj_pos_pb[i, :] + noise_obj[i, :]

                    if self.OBS_KF:
                        obj_pos = obj_pos_kf
                        obj_vel = obj_vel_kf
                    else:
                        obj_pos = obj_pos_pb
                        obj_vel = obj_vel_pb

            for j in range(self.NUM_OBJECTS):
                obj_pos[j, :] = obj_pos[j, :] - drone_pos[0, :]
                
                if self.SRT:
                    obj_pos[i, 0:3] = self.rotate_vector_to_x_axis(obj_pos[i, 0:3])
                    obj_pos_return = np.delete(obj_pos[i, :], 1)
                else:
                    obj_pos_return = np.copy(obj_pos)

            # Compute goal observation
            goal_pos = np.zeros((1, 3))
            goal_pos[0,:] = self.GOAL_POS - drone_pos[0,:]

            # Compute timestep observation
            timestep = np.array([self.step_counter])

            # Compute action difference observation
            self.action_difference = abs(self.curr_action - self.prev_action)
            self.prev_action = np.copy(self.curr_action)
            if self.ACTION_TYPE == ActionType.VEL:
                self.action_difference[0][3] = self.action_difference[0][3]/self.VELOCITY_SIZE
            
            # If using SRT, rotate all other angle dependent states
            if self.SRT:
                drone_vel = self.rotate_3D_vector(drone_vel)
                drone_rpy = self.rotate_euler_angles(drone_rpy)
                drone_rpy_vel = self.rotate_3D_vector(drone_rpy_vel)
                goal_pos = self.rotate_3D_vector(goal_pos)
            '''
            drone_rpy_sin = np.sin(drone_rpy)
            drone_rpy_cos = np.cos(drone_rpy)
            drone_rpy = np.concatenate([drone_rpy_sin, drone_rpy_cos])
            '''
            obs_dict = {
                    "Drone_velocity": drone_vel.flatten(),
                    "Drone_rpy": drone_rpy.flatten(),
                    "Drone_rpy_velocity": drone_rpy_vel.flatten(),
                    "Drone_altitude": drone_altitude.flatten(),
                    "Goal_position": goal_pos.flatten(),
                    "Object_position": obj_pos_return.flatten(),
                    "Object_position_t-1": self.prev_obj_pos.flatten()
                }

            if self.OBS_TIMESTEP:
                obs_dict["Timestep"] = timestep.flatten()

            if self.OBS_OBJ_VEL:
                obs_dict["Object_velocity"] = obj_vel.flatten()

            if self.OBS_ACTION_DIFFERENCE:
                obs_dict["Action_difference"] = abs(self.action_difference).flatten()

            # Store current object positions for next step
            self.prev_obj_pos = obj_pos_return

            return obs_dict

        else:
            return super()._computeObs()

    def _computeInfo(self):
        success = False
        if getattr(self, "truncation_reason", None) == "time_limit":
            if (self.goal_distance <= self.GOAL_RADIUS):
                success = True
        '''
        big_yaw_detected = False
        if abs(self.curr_drone_rpy[2]) > 0.9:
            big_yaw_detected = True
        '''
        info = {"is_success": success,
                "min_object_distance": getattr(self, "min_obj_distance" ,np.nan),
                "max_goal_distance": getattr(self, "max_goal_distance", np.nan),
                "final_drone_altitude": self.curr_drone_pos[2],
                "final_goal_distance": self.goal_distance,
                "out_of_bounds": self.truncation_reason == "out_of_bounds",
                "orientation_out_of_bounds": self.truncation_reason == "orientation",
                "time_limit_reached": self.time_limit_reached,
                "obj_collision": self.collision_type == "object_collision",
                "contact_collision": self.collision_type == "contact_collision",
                "max_kf_pos_error": getattr(self, "max_kf_pos_error", np.nan),
                "average_kf_vel_error": getattr(self, "average_kf_vel_error", np.nan),
                "reward_rpy": self.reward_rpy,
                "reward_goal_distance": self.reward_goal_distance,
                "reward_object_distance": self.reward_object_distance,
                "reward_object_distance_delta": self.reward_object_distance_delta,
                "reward_action_difference": self.reward_action_difference,
                #"big_yaw_detected": big_yaw_detected
                }

        return info  

    def reset_drone(self):
        self.pos = np.zeros((1, 3))
        self.quat = np.zeros((1, 4))
        self.quat[0,3] = 1
        self.rpy = np.zeros((1, 3))
        self.vel = np.zeros((1, 3))
        self.curr_ang_vel = np.zeros((1, 3))

        pb.resetBasePositionAndOrientation(
            self.DRONE_IDS[self.DRONE_ID],
            self.INITIAL_XYZS[self.DRONE_ID,:],
            pb.getQuaternionFromEuler(self.INIT_RPYS[self.DRONE_ID,:]),
            physicsClientId=self.CLIENT)
        pb.resetBaseVelocity(
            self.DRONE_IDS[self.DRONE_ID],
            self.vel[self.DRONE_ID,:],
            self.curr_ang_vel[self.DRONE_ID,:],
            physicsClientId=self.CLIENT)

    def addBall(self,position : None|np.ndarray[float] = None,velocity : None|np.ndarray[float] = None):
        # position: where the ball will be added
        # force: the force applied to the ball at the moment of creation

        if position is None:
            position = np.zeros(3, dtype=float)  # Default position at the origin
        if velocity is None:
            velocity = np.zeros(3, dtype=float)  # Default force is zero
        search_path = os.getcwd() + "/Training/resources"
        pb.setAdditionalSearchPath(search_path)
        
        while (len(self.ball_list) >= self.NUM_OBJECTS):
            # Remove the oldest ball if the limit is reached
            try:
                pb.removeBody(self.ball_list[0], physicsClientId=self.CLIENT)
                self.ball_list.pop(0)
            except Exception:
                pass
                
        print(os.getcwd())

        self.ball_list.append(pb.loadURDF("custom_sphere_small.urdf",
                       basePosition=(position[0], position[1], position[2]),
                       physicsClientId=self.CLIENT))

        pb.resetBaseVelocity(
            objectUniqueId = self.ball_list[-1],
            linearVelocity = velocity,
            physicsClientId = self.CLIENT
        )
    
    def addBallRandom(self):
        # Randomly generate position and force for the ball

        # Generate a random position for the ball
        pos = self.getRandomPos()

        # Calculate force based on the position
        x0, y0, z0 = pos
        target = self.GOAL_POS
        T = np.random.uniform(0.75,1.5) # Time to reach the target, 0.5s minimum

        # Calculate required velocities
        vx = (target[0,0] - x0) / T
        vy = (target[0,1] - y0) / T
        vz = (target[0,2] - z0 + self.G * T**2 / 2) / T

        vx += np.random.uniform(-self.VELOCITY_VAR, self.VELOCITY_VAR)
        vy += np.random.uniform(-self.VELOCITY_VAR, self.VELOCITY_VAR)
        vz += np.random.uniform(-self.VELOCITY_VAR, self.VELOCITY_VAR)

        velocity = np.array([vx, vy, vz]) * (1 + T/13)

        # Add the ball with the generated position and force
        self.addBall(position = pos, velocity = velocity)

    def getRandomPos(self):
        # Generate a random position
        angle = np.random.uniform(-math.pi, math.pi)
        magnitude = np.random.uniform(2, 4)
        x_pos = magnitude * math.cos(angle)
        y_pos = magnitude * math.sin(angle)
        z_pos = np.random.uniform(0.75, 1.75)

        position = x_pos, y_pos, z_pos
        
        return position
    
    def rotate_vector_to_x_axis(self, v):
        v = np.array(v)

        self.rot_angle = np.arctan2(v[1], v[0])

        cos_t = np.cos(-self.rot_angle)
        sin_t = np.sin(-self.rot_angle)
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t,  0],
            [0,     0,      1]
        ])

        rotated = R @ v

        if abs(rotated[1]) < 1e-10:
                        rotated[1] = 0

        return rotated
    
    def reverse_rotate_vector_from_x_axis(self, v):
        v = np.array(v).reshape(3)

        cos_t = np.cos(self.rot_angle)
        sin_t = np.sin(self.rot_angle)
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t,  0],
            [0,     0,      1]
        ])

        return R @ v
    
    def rotate_3D_vector(self, v):
        v = np.array(v).reshape(3)

        cos_t = np.cos(-self.rot_angle)
        sin_t = np.sin(-self.rot_angle)
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t,  0],
            [0,     0,      1]
        ])

        return R @ v
    
    def rotate_euler_angles(self, angles):
        cos_t = np.cos(-self.rot_angle)
        sin_t = np.sin(-self.rot_angle)
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t,  0],
            [0,     0,      1]
        ])

        R_quaternion = Rotation.from_matrix(R).as_quat()
        angles_quaternion = pb.getQuaternionFromEuler(angles.flatten())
        _, new_quaternion = pb.multiplyTransforms([0, 0, 0], R_quaternion, [0, 0, 0], angles_quaternion)
        
        return np.array([pb.getEulerFromQuaternion(new_quaternion)])