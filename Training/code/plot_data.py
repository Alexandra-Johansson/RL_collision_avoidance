from data_handling import Plot

model_type = "PPO"  # "PPO" or "DDPG"

model_filepath_mean_std = "final_models/ActionTypes"
model_filepath_graphs = "final_models/ActionTypes/PPO_PID_2"
model_filepath1 = "final_models/ActionTypes/PPO_VEL_1"
model_filepath2 = "final_models/ActionTypes/PPO_VEL_2"
model_filepath3 = "final_models/ActionTypes/PPO_VEL_3"
model_filepath4 = "final_models/ActionTypes/PPO_VEL_4"
model_filepath5 = "final_models/ActionTypes/PPO_VEL_5"

plot_mean_std = Plot(model_filepath_mean_std, model_type)
plot_graphs = Plot(model_filepath_graphs, model_type)
plot1 = Plot(model_filepath1, model_type)
plot2 = Plot(model_filepath2, model_type)
plot3 = Plot(model_filepath3, model_type)
plot4 = Plot(model_filepath4, model_type)
plot5 = Plot(model_filepath5, model_type)

#plot1.tensorboard()
#plot2.tensorboard()
#plot3.tensorboard()
#plot4.tensorboard()
#plot5.tensorboard()

#plot_graphs.plot_graphs()

file_names = ["PPO_PID", "PPO_VEL"]
number_of_files_each = 5
plot_mean_std.plot_mean_and_std(file_names, number_of_files_each)

input("Press enter to continue...")
