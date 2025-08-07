import os
import numpy as np
import matplotlib.pyplot as plt
from gym_pybullet_drones.utils.enums import DroneModel, ActionType
import torch
from tensorboard import program

class Txt_file:
    def __init__(self,store_path):
        self.store_path = store_path

        self.title = 'PARAMETERS'

    def save_parameters(self, par):
        file_path = self.store_path + f"/parameters.txt"

        with open(file_path, 'w') as file:
            file.write(f"{self.title}\n\n\n")
            for key, value in par.items():
                file.write(f"{key}: {value}\n")

    def load_parameters(self):
        file_path = self.store_path + f"/parameters.txt"

        param = {}
        with open(file_path,'r') as file:
            lines = file.readlines()
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        if (key == "initial_xyzs") or (key == "target_pos"):
                            value = value.replace('[','').replace(']','')
                            value = np.array([np.fromstring(value,sep=' ')])
                        else:
                            value = eval(value)
                    except Exception:
                        print("Error loading parameter: ", key)
                        pass
                    param[key] = value
        return param

class Plot:
    def __init__(self, filename):
        self.filename = filename

        self.plot_path = os.path.join(self.filename, 'plots')
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path + '/')

    def plotReward(self):
        filename = os.path.join(self.filename, 'logs/evaluations.npz')     

        data = np.load(filename)

        steps = data['timesteps']
        rewards = data['results']

        avg_rewards = rewards.mean(axis=1)

        window = 10

        if window > len(steps):
            window = len(steps)

        moving_avg_rewards = np.convolve(avg_rewards, np.ones(window)/window, mode='same')

        steps_moving_avg = steps[:len(moving_avg_rewards)]

        print(np.max(avg_rewards))

        plt.figure(figsize=(10, 6))
        plt.plot(steps, avg_rewards, label="Return", color="blue", alpha=0.4)
        plt.plot(steps_moving_avg, moving_avg_rewards, label=f"{window} SMA Return ", color="red")

        plt.title("Return vs Steps PPO")
        plt.xlabel("Steps")
        plt.ylabel("Return")
        #plt.xlim(0, 1.05*np.max(steps))
        #plt.ylim(-1e4,1.5*np.max(rewards))
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plot_path+"/Return.png", dpi=1000)
        plt.show()

    def tensorboard(self):
        tensorboard_filename = os.path.join(self.filename, 'tensorboard_logs/PPO_1')
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tensorboard_filename])
        url = tb.launch()
        print(f"TensorBoard started at {url}")
