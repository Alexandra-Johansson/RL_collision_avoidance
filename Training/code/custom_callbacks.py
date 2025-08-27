import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_infos = []
        self.reward_infos = []

    def _on_step(self) -> bool:
        # Check if an episode ended
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos,dones):
            if info is not None:
                self.reward_infos.append(info)
            if done and info is not None:
                self.episode_infos.append(info)
                
        return True

    def _on_rollout_end(self):
        # Log all episode infos collected during the rollout
        min_obj_distances = [
            info["min_object_distance"]
            for info in self.episode_infos
            if info is not None and "min_object_distance" in info
        ]
        if min_obj_distances:
            mean_min_obj_distance = np.mean(min_obj_distances)
            self.logger.record("custom/mean_min_object_distance", mean_min_obj_distance)

        max_target_distances = [
            info["max_target_distance"]
            for info in self.episode_infos
            if info is not None and "max_target_distance" in info
        ]
        if max_target_distances:
            mean_max_target_distance = np.mean(max_target_distances)
            self.logger.record("custom/mean_max_target_distance", mean_max_target_distance)

        final_drone_altitudes = [
            info["final_drone_altitude"]
            for info in self.episode_infos
            if info is not None and "final_drone_altitude" in info
        ]
        if final_drone_altitudes:
            mean_final_drone_altitude = np.mean(final_drone_altitudes)
            self.logger.record("custom/mean_final_drone_altitude", mean_final_drone_altitude)

        final_target_distances = [
            info["final_target_distance"]
            for info in self.episode_infos
            if info is not None and "final_target_distance" in info
        ]
        if final_target_distances:
            mean_final_target_distance = np.mean(final_target_distances)
            self.logger.record("custom/mean_final_target_distance", mean_final_target_distance)

        out_of_bounds = [
            info["out_of_bounds"]
            for info in self.episode_infos
            if info is not None and "out_of_bounds" in info
        ]
        if out_of_bounds:
            mean_out_of_bounds = np.mean([int(ob) for ob in out_of_bounds])
            self.logger.record("custom/mean_out_of_bounds", mean_out_of_bounds)

        orientation_out_of_bounds = [
            info["orientation_out_of_bounds"]
            for info in self.episode_infos
            if info is not None and "orientation_out_of_bounds" in info
        ]
        if orientation_out_of_bounds:
            mean_orientation_out_of_bounds = np.mean([int(oob) for oob in orientation_out_of_bounds])
            self.logger.record("custom/mean_orientation_out_of_bounds", mean_orientation_out_of_bounds)

        time_limit_reached = [
            info["time_limit_reached"]
            for info in self.episode_infos
            if info is not None and "time_limit_reached" in info
        ]
        if time_limit_reached:
            mean_time_limit_reached = np.mean([int(tlr) for tlr in time_limit_reached])
            self.logger.record("custom/mean_time_limit_reached", mean_time_limit_reached)

        object_collisions = [
            info["obj_collision"]
            for info in self.episode_infos
            if info is not None and "obj_collision" in info
        ]
        if object_collisions:
            mean_object_collision = np.mean([int(oc) for oc in object_collisions])
            self.logger.record("custom/mean_object_collision", mean_object_collision)

        contact_collisions = [
            info["contact_collision"]
            for info in self.episode_infos
            if info is not None and "contact_collision" in info
        ]
        if contact_collisions:
            mean_contact_collision = np.mean([int(cc) for cc in contact_collisions])
            self.logger.record("custom/mean_contact_collision", mean_contact_collision)

        max_kf_pos_errors = [
            info["max_kf_pos_error"]
            for info in self.episode_infos
            if info is not None and "max_kf_pos_error" in info
        ]
        if max_kf_pos_errors:
            mean_max_kf_pos_error = np.mean(max_kf_pos_errors)
            self.logger.record("custom/mean_max_kf_pos_error", mean_max_kf_pos_error)

        max_kf_vel_errors = [
            info["average_kf_vel_error"]
            for info in self.episode_infos
            if info is not None and "average_kf_vel_error" in info
        ]
        if max_kf_vel_errors:
            mean_max_kf_vel_error = np.mean(max_kf_vel_errors)
            self.logger.record("custom/mean_average_kf_vel_error", mean_max_kf_vel_error)

        reward_rpys = [
            info["reward_rpy"]
            for info in self.reward_infos
            if info is not None and "reward_rpy" in info
        ]
        if reward_rpys:
            mean_reward_rpy = np.mean(reward_rpys)
            self.logger.record("custom/mean_reward_rpy", mean_reward_rpy)

        reward_in_targets = [
            info["reward_in_target"]
            for info in self.reward_infos
            if info is not None and "reward_in_target" in info
        ]
        if reward_in_targets:
            mean_reward_in_target = np.mean(reward_in_targets)
            self.logger.record("custom/mean_reward_in_target", mean_reward_in_target)
        
        reward_target_distances = [
            info["reward_target_distance"]
            for info in self.reward_infos
            if info is not None and "reward_target_distance" in info
        ]
        if reward_target_distances:
            mean_reward_target_distance = np.mean(reward_target_distances)
            self.logger.record("custom/mean_reward_target_distance", mean_reward_target_distance)

        reward_object_distances = [
            info["reward_object_distance"]
            for info in self.reward_infos
            if info is not None and "reward_object_distance" in info
        ]
        if reward_object_distances:
            mean_reward_object_distance = np.mean(reward_object_distances)
            self.logger.record("custom/mean_reward_object_distance", mean_reward_object_distance)

        reward_object_distance_deltas = [
            info["reward_object_distance_delta"]
            for info in self.reward_infos
            if info is not None and "reward_object_distance_delta" in info
        ]
        if reward_object_distance_deltas:
            mean_reward_object_distance_delta = np.mean(reward_object_distance_deltas)
            self.logger.record("custom/mean_reward_object_distance_delta", mean_reward_object_distance_delta)
        
        reward_action_differences = [
            info["reward_action_difference"]
            for info in self.reward_infos
            if info is not None and "reward_action_difference" in info
        ]
        if reward_action_differences:
            mean_reward_action_difference = np.mean(reward_action_differences)
            self.logger.record("custom/mean_reward_action_difference", mean_reward_action_difference)

        big_yaw_detecteds = [
            info["big_yaw_detected"]
            for info in self.reward_infos
            if info is not None and "big_yaw_detected" in info
        ]
        if big_yaw_detecteds:
            mean_big_yaw_detected = np.mean([int(byd) for byd in big_yaw_detecteds])
            self.logger.record("custom/mean_big_yaw_detected", mean_big_yaw_detected)
        
        self.episode_infos = [] # Resetting episode/rollout info
        self.reward_infos = []

        return True

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, vec_env, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.save_dir = save_dir
    
    def _on_step(self):
        return True

    def _on_event(self):
        path = os.path.join(self.save_dir, "/best_vecnormalize.pkl")
        self.vec_env.save(path)
        if self.verbose > 0:
            print(f"Saved VecNormaliz to {path}")