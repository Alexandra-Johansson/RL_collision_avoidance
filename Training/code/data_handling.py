import os
import numpy as np
import matplotlib.pyplot as plt
from gym_pybullet_drones.utils.enums import DroneModel, ActionType
from tensorboard import program
from tensorboard.backend.event_processing import event_accumulator

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
                        if (key == "initial_xyzs") or (key == "goal_pos"):
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
    def __init__(self, filename, model_type):
        self.filename = filename
        self.model_type = model_type

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
        tensorboard_filename = os.path.join(self.filename, 'tensorboard_logs/' + self.model_type + '_1')
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tensorboard_filename])
        url = tb.launch()
        print(f"TensorBoard started at {url}")

    def plot_graphs(self):
        log_filename = os.path.join(self.filename, 'tensorboard_logs/' + self.model_type + '_1')

        plot_tags = ['eval/mean_reward', 'eval/success_rate', 'rollout/ep_rew_mean', 'rollout/success_rate', 'custom/mean_contact_collision', 
             'custom/mean_object_collision', 'custom/mean_final_drone_altitude', 'custom/mean_final_goal_distance',
             'custom/mean_max_goal_distance', 'custom/mean_min_object_distance', 'custom/mean_reward_action_difference',
             'custom/mean_reward_object_distance', 'custom/mean_reward_goal_distance']
        
        ea = event_accumulator.EventAccumulator(log_filename)
        ea.Reload()

        scalar_tags = ea.Tags()['scalars']

        for tag in scalar_tags:
            if tag in plot_tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                plt.figure()
                plt.plot(steps, values, label=tag)
                plt.xlabel('Step')
                if 'altitude' in tag or 'distance' in tag:
                    plt.ylabel('Meters')
                elif 'reward' in tag or 'rew' in tag:
                    plt.ylabel('Reward')
                elif 'collision' in tag or 'rate' in tag:
                    plt.ylabel('Rate')
                else:
                    plt.ylabel('Value')
                plt.title(f'{tag}')
                plt.grid(True)
                plt.savefig(self.plot_path + f"/{tag.replace('/', '_')}_plot.png")
                plt.close()

    def plot_mean_and_std(self, file_names, number_of_files_each):
        for file_name in file_names:
            for i in range(number_of_files_each):
                log_filename = os.path.join(self.filename, file_name + f'_{i+1}' + '/tensorboard_logs/' + self.model_type + '_1')
                
                plot_tags = ['rollout/ep_rew_mean', 'rollout/success_rate']
        
                ea = event_accumulator.EventAccumulator(log_filename)
                ea.Reload()

                scalar_tags = ea.Tags()['scalars']

                for tag in scalar_tags:
                    if tag in plot_tags:
                        events = ea.Scalars(tag)
                        try:
                            total_reward
                        except NameError:
                            steps = [e.step for e in events]
                            total_reward = np.zeros((len(steps),number_of_files_each))
                            total_success = np.zeros((len(steps),number_of_files_each))
                        values = [e.value for e in events]
                        if 'rew' in tag:
                            total_reward[:, i] = values
                        elif 'success' in tag:
                            total_success[:, i] = values

            # Plot reward mean with std
            reward_mean = np.mean(total_reward,1)
            reward_std = np.std(total_reward,1)
            
            plt.figure()
            plt.plot(steps, reward_mean, label=tag)
            plt.fill_between(steps, reward_mean-reward_std, reward_mean+reward_std, alpha = 0.2)
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title(f'Reward over {number_of_files_each} models with ' + file_name)
            plt.grid(True)
            plt.savefig(self.plot_path + "/" + file_name + "_reward_plot.png")
            final_25_percent_mean = [mean for (step, mean) in zip(steps, reward_mean) if step > 1.5*1e6]
            final_25_percent_std = [std for (step, std) in zip(steps, reward_std) if step > 1.5*1e6]
            print("Final reward mean: " + f"{np.mean(final_25_percent_mean)}")
            print("Final reward std: " + f"{np.mean(final_25_percent_std)}")
            plt.close()

            # Plot success rate mean with std
            success_mean = np.mean(total_success,1)
            success_std = np.std(total_success,1)

            plt.figure()
            plt.plot(steps, success_mean, label=tag)
            plt.fill_between(steps, success_mean-success_std, success_mean+success_std, alpha = 0.2)
            plt.xlabel('Step')
            plt.ylabel('Success rate')
            plt.title(f'Success over {number_of_files_each} models with ' + file_name)
            plt.grid(True)
            plt.savefig(self.plot_path + "/" + file_name + "_success_plot.png")
            final_25_percent_mean = [mean for (step, mean) in zip(steps, success_mean) if step > 1.5*1e6]
            final_25_percent_std = [std for (step, std) in zip(steps, success_std) if step > 1.5*1e6]
            print("Final success mean: " + f"{np.mean(final_25_percent_mean)}")
            print("Final success std: " + f"{np.mean(final_25_percent_std)}")
            plt.close

            success_rates = [success for (step, success) in zip(steps, total_success) if step > 1.75*1e6]
            success_rates_mean = np.mean(success_rates,0)
            best_model_index = np.argmax(success_rates_mean)
            print("Best model index " + f"{best_model_index + 1}" + " with success rate of " + f"{success_rates_mean[best_model_index]}")