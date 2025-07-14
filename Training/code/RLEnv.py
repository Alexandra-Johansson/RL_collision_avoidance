import math
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


class RLEnv(BaseRLAviary):
    def __init__(self,
                 num_drones = 1,
                 neighborhood_radius = np.inf,
                 initial_rpys = None,
                 physics = Physics.PYB,
                 gui = False,
                 record = False,
                 obs = ObservationType.KIN,
                 act = ActionType.PID,
                 parameters = None):

        self.DRONE_MODEL = parameters['drone_model']
        self.NUM_OBJECTS = parameters['num_objects']
        self.INITIAL_XYZS = parameters['initial_xyzs']
        self.CTRL_FREQ = parameters['ctrl_freq']
        self.PYB_FREQ = parameters['pyb_freq']
        self.ACTION_SIZE = parameters['action_size']
        self.TARGET_POS = parameters['target_pos']
        self.TARGET_RADIUS = parameters['target_radius']
        self.AVOIDANCE_RADIUS = parameters['avoidance_radius']
        self.CRITICAL_SAFETY_DISTANCE = parameters['critical_safety_distance']
        self.OBS_NOISE = parameters['obs_noise']
        self.OBS_NOISE_STD = parameters['obs_noise_std']
        self.PROCESS_NOISE = parameters['kf_process_noise']
        self.MEASUREMENT_NOISE = parameters['kf_measurement_noise']

        self.REWARD_COLLISION = parameters['reward_collision']
        self.REWARD_TERMINATED = parameters['reward_terminated']
        self.REWARD_TARGET_DISTANCE = parameters['reward_target_distance']
        self.REWARD_TARGET_DISTANCE_DELTA = parameters['reward_target_distance_delta']
        self.REWARD_ANGULAR_VELOCITY_DELTA = parameters['reward_angular_velocity_delta']
        self.REWARD_OBJECT_DISTANCE_DELTA = parameters['reward_object_distance_delta']
        self.REWARD_STEP = parameters['reward_step']
        self.REWARD_IN_TARGET = parameters['reward_in_target']

        self.ORIGINAL_XYZS = np.copy(self.INITIAL_XYZS)

        self.VELOCITY_VAR = 0

        self.ball_list = []
        self.collision_detected = False
        
        super().__init__(self.DRONE_MODEL,
                         num_drones,
                         neighborhood_radius,
                         self.INITIAL_XYZS,
                         initial_rpys,
                         physics,
                         self.PYB_FREQ,
                         self.CTRL_FREQ,
                         gui,
                         record,
                         obs,
                         act)

        self.EPISODE_LEN_SEC = parameters['episode_length']

        # Add more filters if multiple objects should be tracked
        self.kf = KalmanFilter(dt = self.CTRL_TIMESTEP, 
                               process_var = self.PROCESS_NOISE,
                               measurement_var = self.MEASUREMENT_NOISE,
                               gravity = self.G)


        drone_id = 0
        drone_state_vec = self._getDroneStateVector(drone_id)

        self.curr_drone_pos = drone_state_vec[0:3]  # Position
        self.target_distance = np.linalg.norm(self.TARGET_POS[0] - drone_state_vec[0:3])
        self.ang_vel = drone_state_vec[13:16]  # Angular velocity
        self.obj_distances = np.zeros((self.NUM_OBJECTS, 1))

    def step(self, rel_action):
        # Convert the scaled relative action to a global action
        action = self.curr_drone_pos + rel_action*self.ACTION_SIZE
        #action = self.TARGET_POS
        obs, reward, terminated, truncated, info = super().step(action)
        # Compute the reward, termination, truncation, and info for one step

        if self.GUI:
            time.sleep(self.CTRL_TIMESTEP)

        if self._getCollision(self.DRONE_IDS[0]):
            reward = self._computeReward()

            self.reset_drone()

            action = np.array([[0,0,0]])

            obs, _, terminated, truncated, info = super().step(action)

            #print(time.time(), ": Collision detected")
            
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

        self.reset_drone()

        obs, info = super().reset(seed, options)

        drone_state_vec = self._getDroneStateVector(drone_id)
        self.curr_drone_pos = drone_state_vec[0:3]  # Position
        self.target_distance = np.linalg.norm(self.TARGET_POS[0] - drone_state_vec[0:3])
        self.ang_vel = drone_state_vec[13:16] # Angular velocity
        self.collision_detected = False

        self.addBallRandom()

        return obs, info

    def _computeReward(self):
        # TODO
        ret = 0.0

        drone_id = 0
        drone_state_vec = self._getDroneStateVector(drone_id)

        self.curr_drone_pos = drone_state_vec[0:3]

        prev_ang_vel = np.copy(self.ang_vel)
        self.ang_vel = drone_state_vec[13:16]

        prev_target_distance = np.copy(self.target_distance)
        self.target_distance = np.linalg.norm(self.TARGET_POS[0] - self.curr_drone_pos)

        prev_obj_distances = np.copy(self.obj_distances)
        self.obj_distances = np.zeros((len(self.ball_list), 1))
        # Calculate the distance to each object
        for i in range(len(self.ball_list)):
            ball_pos, _ = pb.getBasePositionAndOrientation(self.ball_list[i], physicsClientId=self.CLIENT)
            self.obj_distances[i] = np.linalg.norm(ball_pos[0:3] - self.curr_drone_pos)

        # Check if the episode is terminated
        if self._computeTerminated():
            if self.collision_detected:
                ret = self.REWARD_COLLISION
                # Negative reward for collision
            else:
                ret = self.REWARD_TERMINATED
                # Reward for avoiding all obstacles
        # Else calculate the reward based on the states
        else:
            obj_distances_delta = np.sum(prev_obj_distances - self.obj_distances)
            
            target_distance_delta = np.sum(prev_target_distance - self.target_distance)

            ang_vel_delta = np.sum(abs(prev_ang_vel - self.ang_vel))

            # Reward based on factors
            #ret += ang_vel_delta*self.REWARD_ANGULAR_VELOCITY_DELTA # Negative reward based on change in velocity
            #ret += -10*np.sum(abs(self.ang_vel))
            #ret += -1000*sum(abs(drone_state_vec[7:8]))

            if (self.target_distance <= self.TARGET_RADIUS):
                # If the drone is within the target radius, give a positive reward
                ret += self.REWARD_IN_TARGET
                #ret += ((self.REWARD_TARGET_DISTANCE_DELTA * target_distance_delta)/self.CTRL_TIMESTEP)/2
            else:
                # If the drone is outside the target radius, give a negative reward
                ret += self.REWARD_TARGET_DISTANCE * self.target_distance
                #ret += (self.REWARD_TARGET_DISTANCE_DELTA * target_distance_delta)/self.CTRL_TIMESTEP
                # Negative reward for each step outside the target
                ret += self.REWARD_STEP
            
            # 2. Negative reward based on if the object is moving towards the drone
            if (obj_distances_delta < 0):
                # If the object is to close, give a reward based on change in distance
                #ret += self.REWARD_OBJECT_DISTANCE_DELTA * obj_distances_delta
                pass

        #print(f"Reward: {ret}")
        return float(ret)

    def _computeTerminated(self):
        Terminated = False
        
        '''
        # If the drone has avoided all obstacles, terminate the episode
        for ball in self.ball_list:
            vel, ang_v = pb.getBaseVelocity(ball, physicsClientId = self.CLIENT)
            pos, quat = pb.getBasePositionAndOrientation(ball, physicsClientId = self.CLIENT)
            if (vel[2] > 1e-4) or (pos[2] > 1e-1):
                # If a ball is in the air don't terminate
                return Terminated
        '''
        if (self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC):
            Terminated = True
            #print("Time limit reached, episode terminated.")
        elif any(self.obj_distances < self.CRITICAL_SAFETY_DISTANCE):
            Terminated = True
            self.collision_detected = True
            #print("Collision detected, Terminating episode")
        ''' 
        if self.target_distance < self.TARGET_RADIUS:
            # If the drone is within the target radius, terminate the episode
            Terminated = True
            #print("Drone within target radius, episode terminated.")

        if Terminated:
            self.getRandomInitialPos()
        '''

        # If no ball is in the air terminate
        #Terminated = True
        return Terminated

    def _computeTruncated(self):
        Truncated = False

        drone_id = 0
        drone_state_vec = self._getDroneStateVector(drone_id)

        if (abs(drone_state_vec[0]) > 5 or
            abs(drone_state_vec[1]) > 5 or
            abs(drone_state_vec[2]) > 5):
            Truncated = True
            #print("Drone position out of bounds, episode truncated.")

        elif (abs(drone_state_vec[7]) > .9 or
                abs(drone_state_vec[8]) > .9):
            Truncated = True
            #print("Drone orientation out of bounds, episode truncated.")

        '''
        if Truncated:
            self.getRandomInitialPos()
        '''
        
        return Truncated

    def _getCollision(self, obj):

        constact_points = pb.getContactPoints(obj, physicsClientId=self.CLIENT)

        if len(constact_points) > 0:
            return True
        else:
            return False

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.KIN and self.ACT_TYPE == ActionType.PID:
            pos_low = np.array([-5.0, -5., -5.])
            pos_high = np.array([5.0, 5.0, 5.0])
            vel_low = np.array([-2,-2,-2])
            vel_high = np.array([2,2,2])
            ball_vel_low = np.array([-20, -20, -20])
            ball_vel_high = np.array([20, 20, 20])
            rpy_low = np.array(-0.9*np.ones(3))
            rpy_high = np.array(0.9*np.ones(3))
            rpy_vel_low = np.array([-10, -10, -6])
            rpy_vel_high = np.array([10, 10, 6])
            target_pos_low = np.array([-5.0, -5.0, -5.0])
            target_pos_high = np.array([5.0, 5.0, 5.0])

            # Add drone observation space
            #obs_drone_lower_bound = np.array([pos_low for drone in range(self.NUM_DRONES)])
            #obs_drone_upper_bound = np.array([pos_high for drone in range(self.NUM_DRONES)])
            obs_drone_vel_lower_bound = np.array([vel_low for drone in range(self.NUM_DRONES)])
            obs_drone_vel_upper_bound = np.array([vel_high for drone in range(self.NUM_DRONES)])
            obs_drone_rpy_lower_bound = np.array([rpy_low for drone in range(self.NUM_DRONES)])
            obs_drone_rpy_upper_bound = np.array([rpy_high for drone in range(self.NUM_DRONES)])
            obs_drone_rpy_vel_lower_bound = np.array([rpy_vel_low for drone in range(self.NUM_DRONES)])
            obs_drone_rpy_vel_upper_bound = np.array([rpy_vel_high for drone in range(self.NUM_DRONES)])

            # Add target observation space
            obs_target_lower_bound = np.array([target_pos_low for drone in range(self.NUM_DRONES)])
            obs_target_upper_bound = np.array([target_pos_high for drone in range(self.NUM_DRONES)])

            # Add object observation space
            obs_obj_lower_bound = np.array([pos_low for obj in range(self.NUM_OBJECTS)])
            obs_obj_upper_bound = np.array([pos_high for obj in range(self.NUM_OBJECTS)])
            obs_obj_vel_lower_bound = np.array([ball_vel_low for obj in range(self.NUM_OBJECTS)])
            obs_obj_vel_upper_bound = np.array([ball_vel_high for obj in range(self.NUM_OBJECTS)])

            '''
            return spaces.Dict({
                #"Drone_position": spaces.Box(low=obs_drone_lower_bound, high=obs_drone_upper_bound, dtype=np.float32),
                "Drone_velocity": spaces.Box(low=obs_drone_vel_lower_bound, high=obs_drone_vel_upper_bound, dtype=np.float32),
                "Drone_rpy": spaces.Box(low=obs_drone_rpy_lower_bound, high=obs_drone_rpy_upper_bound, dtype=np.float32),
                "Drone_rpy_velocity": spaces.Box(low=obs_drone_rpy_vel_lower_bound, high=obs_drone_rpy_vel_upper_bound, dtype=np.float32),
                "Target_distance": spaces.Box(low=obs_target_lower_bound, high=obs_target_upper_bound, dtype=np.float32),
                "Object_position": spaces.Box(low=obs_obj_lower_bound, high=obs_obj_upper_bound, dtype=np.float32)})
                #"Object_velocity": spaces.Box(low=obs_obj_vel_lower_bound, high=obs_obj_vel_upper_bound, dtype=np.float32)})
            '''
            return spaces.Dict({
                #"Drone_position": spaces.Box(low=obs_drone_lower_bound, high=obs_drone_upper_bound, dtype=np.float32),
                "Drone_velocity": spaces.Box(low=obs_drone_vel_lower_bound.flatten(), high=obs_drone_vel_upper_bound.flatten(), dtype=np.float64),
                "Drone_rpy": spaces.Box(low=obs_drone_rpy_lower_bound.flatten(), high=obs_drone_rpy_upper_bound.flatten(), dtype=np.float64),
                "Drone_rpy_velocity": spaces.Box(low=obs_drone_rpy_vel_lower_bound.flatten(), high=obs_drone_rpy_vel_upper_bound.flatten(), dtype=np.float64),
                "Target_distance": spaces.Box(low=obs_target_lower_bound.flatten(), high=obs_target_upper_bound.flatten(), dtype=np.float64),
                "Object_position": spaces.Box(low=obs_obj_lower_bound.flatten(), high=obs_obj_upper_bound.flatten(), dtype=np.float64)})
                #"Object_velocity": spaces.Box(low=obs_obj_vel_lower_bound, high=obs_obj_vel_upper_bound, dtype=np.float32)})

    
        else:
            super()._observationSpace()

    def _actionSpace(self):
        '''
        size = 3
        act_lower_bound = np.array([-self.ACTION_SIZE*np.ones(size)])
        act_lower_bound[0,2] = act_lower_bound[0,2]/10
        act_upper_bound = np.array([ self.ACTION_SIZE*np.ones(size)])
        act_upper_bound[0,2] = act_upper_bound[0,2]*2

        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        '''
        return super()._actionSpace()
    
    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.KIN and self.ACT_TYPE == ActionType.PID:
            if self.OBS_NOISE:
                #noise_drone = np.random.normal(0, self.OBS_NOISE_STD, (self.NUM_DRONES, 3))
                noise_obj = np.random.normal(0, self.OBS_NOISE_STD, (self.NUM_OBJECTS, 3))
            else:
                #noise_drone = np.zeros((self.NUM_DRONES, 3))
                noise_obj = np.zeros((self.NUM_OBJECTS, 3))

            # Compute drone observations
            drone_pos = np.zeros((self.NUM_DRONES, 3))
            drone_vel = np.zeros((self.NUM_DRONES, 3))
            drone_rpy = np.zeros((self.NUM_DRONES, 3))
            drone_rpy_vel = np.zeros((self.NUM_DRONES, 3))
            for i in range(self.NUM_DRONES):
                obs = self._getDroneStateVector(i)
                drone_pos[i,:] = obs[0:3]
                drone_vel[i,:] = obs[10:13]
                drone_rpy[i,:] = obs[7:10]
                drone_rpy_vel[i,:] = obs[13:16]

            # Compute object observations
            obj_pos = np.zeros((self.NUM_OBJECTS, 3))
            obj_vel = np.zeros((self.NUM_OBJECTS, 3))
            for i in range(self.NUM_OBJECTS):
                if i < len(self.ball_list):
                    obs, _ = pb.getBasePositionAndOrientation(self.ball_list[i], physicsClientId=self.CLIENT)
                    self.kf.measurementUpdate(obs[0:3]) #+ noise_obj[i, :])
                    obj_temp = self.kf.getState()
                    obj_pos[i, :] = obj_temp[0:3, 0]
                    obj_vel[i, :] = obj_temp[3:6, 0]

            for i in range(self.NUM_DRONES):
                for j in range(self.NUM_OBJECTS):
                    obj_pos[j, :] = obj_pos[j, :] - drone_pos[i, :]

            # Compute target observation
            target_distance = np.zeros((self.NUM_DRONES, 3))
            for i in range(self.NUM_DRONES):
                target_distance[i,:] = self.TARGET_POS - drone_pos[i,:]
            

            obs_dict = {
                #"Drone_position": drone_pos,
                "Drone_velocity": drone_vel.flatten(),
                "Drone_rpy": drone_rpy.flatten(),
                "Drone_rpy_velocity": drone_rpy_vel.flatten(),
                "Target_distance": target_distance.flatten(),
                "Object_position": obj_pos.flatten()
                #"Object_velocity": obj_vel
            }
            return obs_dict

        else:
            return super()._computeObs()

    def _computeInfo(self):
        # TODO
        success = False
        if self._computeTerminated:
            if not self.collision_detected:
                success = True

        info = {"info": 0,
                "is_success": success}
        
        return info  

    def reset_drone(self):
        drone_id = 0

        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.quat[0,3] = 1
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_vel = np.zeros((self.NUM_DRONES, 3))

        #self.getRandomInitialPos()

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

    def addBall(self,position : None|np.ndarray[float] = None,velocity : None|np.ndarray[float] = None):
        # position: where the ball will be added
        # force: the force applied to the ball at the moment of creation

        if position is None:
            position = np.zeros(3, dtype=float)  # Default position at the origin
        if velocity is None:
            velocity = np.zeros(3, dtype=float)  # Default force is zero
        search_path = "Training/resources"
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
        target = self.TARGET_POS
        T = np.random.uniform(0.5,1.5) # Time to reach the target

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
        magnitude = np.random.uniform(2, 3)
        x_pos = magnitude * math.cos(angle)
        y_pos = magnitude * math.sin(angle)
        z_pos = np.random.uniform(0.75, 1.25)
        z_pos = 1.0

        position = [x_pos, y_pos, z_pos]
        
        return position
    
    def getRandomInitialPos(self):
        pos = self.getRandomPos()
        self.INIT_XYZS[0,0] = pos[0]
        self.INIT_XYZS[0,1] = pos[1]
        self.INIT_XYZS[0,2] = pos[2]