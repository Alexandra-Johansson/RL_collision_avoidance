from data_handling import Plot

model_type = "PPO"  # "PPO" or "DDPG"

model_filepath_mean_std = "final_models/ActionTypes"
model_filepath_graphs = "final_models/PPO_95%"
model_filepath1 = "final_models/PPO_95%"
model_filepath2 = "final_models/PPO_good"
model_filepath3 = "final_models/PPO_ok_1"
model_filepath4 = "final_models/PPO_ok_2"

plot_mean_std = Plot(model_filepath_mean_std, model_type)
plot_graphs = Plot(model_filepath_graphs, model_type)
plot1 = Plot(model_filepath1, model_type)
plot2 = Plot(model_filepath2, model_type)
plot3 = Plot(model_filepath3, model_type)
plot4 = Plot(model_filepath4, model_type)

#plot1.tensorboard()
#plot2.tensorboard()
#plot3.tensorboard()
#plot4.tensorboard()

#plot_mean_std.plot_graphs()

file_names = ["PPO_PID", "PPO_VEL"]
number_of_files_each = 5
plot_mean_std.plot_mean_and_std(file_names, number_of_files_each)

input("Press enter to continue...")
