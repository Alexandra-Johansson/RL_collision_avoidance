import numpy as np
import math

import pybullet as pb
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
        self.TARGET_RADIUS = parameters['target_radius']
        self.AVOIDANCE_RADIUS = parameters['avoidance_radius']
        self.OBS_NOISE = parameters['obs_noise']
        self.OBS_NOISE_STD = parameters['obs_noise_std']

        self.REWARD_COLLISION = parameters['reward_collision']
        self.REWARD_TERMINATED = parameters['reward_terminated']
        self.REWARD_TARGET_DISTANCE = parameters['reward_target_distance']
        self.REWARD_TARGET_DISTANCE_DELTA = parameters['reward_target_distance_delta']
        self.REWARD_ANGULAR_VELOCITY_DELTA = parameters['reward_angular_velocity_delta']
        self.REWARD_OBJECT_DISTANCE_DELTA = parameters['reward_object_distance_delta']
        self.REWARD_STEP = parameters['reward_step']
        self.REWARD_IN_TARGET = parameters['reward_in_target']

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

        self.curr_drone_pos = drone_state_vec[0:3]  # Position
        self.target_distance = np.linalg.norm(self.TARGET_POS[0] - drone_state_vec[0:3])
        self.ang_vel = drone_state_vec[13:16]  # Angular velocity
        self.obj_distances = np.zeros((self.NUM_OBJECTS, 1))

    def step(self, rel_action):
        action = self.curr_drone_pos + rel_action
        #print(f"Action: {action}")
        #action = np.array([[.0,.0,-1.0]])
        obs, reward, terminated, truncated, info = super().step(action)
        # Compute the reward, termination, truncation, and info for one step

        if self._getCollision(self.DRONE_IDS[0]):
            reward = self._computeReward()

            self.reset_drone()

            action = np.array([[0,0,0]])

            obs, _, terminated, truncated, info = super().step(action)
            
        return obs, reward, terminated, truncated, info

    def reset(self, seed = None, options = None):
        drone_id = 0
        self.ctrl[drone_id].reset()
        
        for ball in self.ball_list:
            try:
                pb.removeBody(ball, physicsClientId=self.CLIENT)
                self.ball_list.pop(0)
            except Exception:
                # If the ball is already removed, ignore the error
                pass

        obs, info = super().reset(seed, options)

        drone_state_vec = self._getDroneStateVector(drone_id)
        self.curr_drone_pos = drone_state_vec[0:3]  # Position
        self.target_distance = np.linalg.norm(self.TARGET_POS[0] - drone_state_vec[0:3])
        self.ang_vel = drone_state_vec[13:16] # Angular velocity

        #self.addBallRandom()

        return obs, info

    def _computeReward(self):
        # TODO
        ret = 0.0

        drone_id = 0
        drone_state_vec = self._getDroneStateVector(drone_id)

        self.curr_drone_pos = drone_state_vec[0:3]
        prev_ang_vel = self.ang_vel
        self.ang_vel = drone_state_vec[13:16]

        prev_target_distance = np.copy(self.target_distance)
        self.target_distance = np.linalg.norm(self.TARGET_POS[0] - self.curr_drone_pos)

        prev_obj_distances = np.copy(self.obj_distances)
        self.obj_distances = np.zeros((len(self.ball_list), 1))
        # Calculate the distance to each object
        for i in range(len(self.ball_list)):
            ball_pos, _ = pb.getBasePositionAndOrientation(self.ball_list[i], physicsClientId=self.CLIENT)
            self.obj_distances[i] = np.linalg.norm(ball_pos[0:3] - self.curr_drone_pos)

        # Check for collisions and compute rewards
        if self._getCollision(self.DRONE_IDS[0]):
            ret = self.REWARD_COLLISION
            # Negative reward for collision
        # Check if the episode is terminated
        elif self._computeTerminated():
            ret = self.REWARD_TERMINATED
            # Reward for avoiding all obstacles
        # Else calculate the reward based on the states
        else:
            obj_distances_delta = prev_obj_distances - self.obj_distances
            
            target_distance_delta = prev_target_distance - self.target_distance

            ang_vel_delta = np.sum((prev_ang_vel - self.ang_vel)**2)

            # Reward based on factors
            ret += ang_vel_delta*self.REWARD_ANGULAR_VELOCITY_DELTA # Negative reward based on change in velocity
            if (self.target_distance <= self.TARGET_RADIUS):
                # If the drone is within the target radius, give a positive reward
                ret += self.REWARD_IN_TARGET
                ret += ((self.REWARD_TARGET_DISTANCE_DELTA * target_distance_delta)/self.CTRL_TIMESTEP)/2
            else:
                # If the drone is outside the target radius, give a negative reward
                ret += self.REWARD_TARGET_DISTANCE * self.target_distance
                ret += (self.REWARD_TARGET_DISTANCE_DELTA * target_distance_delta)/self.CTRL_TIMESTEP
            # 2. Negative reward based on if objects are closer or further away
            if (self.obj_distances < self.AVOIDANCE_RADIUS):
                # If the object is to close, give a reward based on change in distance
                ret += self.REWARD_OBJECT_DISTANCE_DELTA * obj_distances_delta
            # 3. Negative reward for each step
            #ret += self.REWARD_STEP

        #print(f"Reward: {ret}")
        return ret

    def _computeTerminated(self):
        Terminated = False
        
        # If the drone has avoided all obstacles, terminate the episode
        for ball in self.ball_list:
            vel, ang_v = pb.getBaseVelocity(ball, physicsClientId = self.CLIENT)
            pos, quat = pb.getBasePositionAndOrientation(ball, physicsClientId = self.CLIENT)
            if (vel[2] > 1e-4) or (pos[2] > 1e-1):
                # If a ball is in the air don't terminate
                return Terminated
            
        if self.target_distance < self.TARGET_RADIUS:
            # If the drone is within the target radius, terminate the episode
            Terminated = True
            print("Drone within target radius, episode terminated.")
        
        # If no ball is in the air terminate
        # Terminated = True
        return Terminated

    def _computeTruncated(self):
        Truncated = False

        drone_id = 0
        drone_state_vec = self._getDroneStateVector(drone_id)

        if (self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC):
            Truncated = True
            print("Time limit reached, episode truncated.")

        elif (self._getCollision(self.DRONE_IDS[0])):
            Truncated = True
            print("Collision detected, episode truncated.")
        
        elif (abs(drone_state_vec[0]) > 5 or
            abs(drone_state_vec[1]) > 5 or
            abs(drone_state_vec[2]) > 5):
            Truncated = True
            print("Drone out of bounds, episode truncated.")

        elif (abs(drone_state_vec[7]) > .9 or
                abs(drone_state_vec[8]) > .9):
            Truncated = True
            print("Drone orientation out of bounds, episode truncated.")

        return Truncated

    def _getCollision(self, obj):

        constact_points = pb.getContactPoints(obj, physicsClientId=self.CLIENT)

        if len(constact_points) > 0:
            return True
        else:
            return False

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.KIN and self.ACT_TYPE == ActionType.PID:
            pos_low = np.array([-5.0, -5.0, 0])
            pos_high = np.array([5.0, 5.0, 5.0])

            # Add drone obesrvation space
            obs_drone_lower_bound = np.array([pos_low for drone in range(self.NUM_DRONES)])
            obs_drone_upper_bound = np.array([pos_high for drone in range(self.NUM_DRONES)])

            # Add object observation space
            obs_obj_lower_bound = np.hstack([np.array([pos_low for obj in range(self.NUM_OBJECTS)])])
            obs_obj_upper_bound = np.hstack([np.array([pos_high for obj in range(self.NUM_OBJECTS)])])

            return spaces.Dict({
                "Drone_position": spaces.Box(low=obs_drone_lower_bound, high=obs_drone_upper_bound, dtype=np.float32),
                "Object_position": spaces.Box(low=obs_obj_lower_bound, high=obs_obj_upper_bound, dtype=np.float32)})
        else:
            super()._observationSpace()

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
            for i in range(len(self.ball_list)):
                obs, _ = pb.getBasePositionAndOrientation(self.ball_list[i], physicsClientId=self.CLIENT)
                obj_pos[i,:] = obs[0:3] + noise_obj[i,:]

            return {
                "Drone_position": drone_pos,
                "Object_position": obj_pos
            }


        else:
            return super()._computeObs()

    def _computeInfo(self):
        # TODO
        return {"info": 0}   

    def reset_drone(self):
        drone_id = 0

        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.quat[0,3] = 1
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_vel = np.zeros((self.NUM_DRONES, 3))

        pb.resetBasePositionAndOrientation(
            self.DRONE_IDS[drone_id],
            self.INITIAL_XYZS[drone_id,:],
            pb.getQuaternionFromEuler(self.INIT_RPYS[drone_id,:]),
            physicsClientId=self.CLIENT)
        pb.resetBaseVelocity(
            self.DRONE_IDS[drone_id],
            self.vel[drone_id,:],
            self.ang_vel[drone_id,:],
            physicsClientId=self.CLIENT)
        # Reset the drone state vector

        self.ctrl[drone_id].reset()

    def addBall(self,position : None|np.ndarray[float] = None,force : None|np.ndarray[float] = None):
        # position: where the ball will be added
        # force: the force applied to the ball at the moment of creation

        if position is None:
            position = np.zeros(3, dtype=float)  # Default position at the origin
        if force is None:
            force = np.zeros(3, dtype=float)  # Default force is zero
        search_path = "/home/alex/Desktop/Exjobb/RL_collision_avoidance/Training/resources"
        pb.setAdditionalSearchPath(search_path)
        
        while (len(self.ball_list) >= self.NUM_OBJECTS):
            # Remove the oldest ball if the limit is reached
            try:
                pb.removeBody(self.ball_list[0], physicsClientId=self.CLIENT)
                self.ball_list.pop(0)
            except Exception:
                pass
                
        self.ball_list.append(pb.loadURDF("custom_sphere_small.urdf",
                       basePosition=(position[0], position[1], position[2]),
                       physicsClientId=self.CLIENT))
        
        position = [0, 0, 0]  # Relative position (center of mass)
        pb.applyExternalForce(
            objectUniqueId=self.ball_list[-1],  # Get the last added ball
            linkIndex=-1,  # -1 means base/root link
            forceObj=force,
            posObj=position,
            flags=pb.WORLD_FRAME
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
    
