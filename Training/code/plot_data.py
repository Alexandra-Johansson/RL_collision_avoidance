from data_handling import Plot

model_type = "DDPG"  # "PPO" or "DDPG"

model_filepath = "Training/results/DDPG_training_2025.08.10-05.45.27"
model_filepath2 = "Training/results/DDPG_training_2025.08.10-15.45.39"
model_filepath3 = "Training/results/DDPG_training_2025.08.10-23.46.47"
model_filepath4 = "Training/results/DDPG_training_2025.08.11-07.41.57"

plot = Plot(model_filepath, model_type)
plot2 = Plot(model_filepath2, model_type)
plot3 = Plot(model_filepath3, model_type)
plot4 = Plot(model_filepath4, model_type)
#plot.plotReward()
plot.tensorboard()
plot2.tensorboard()
plot3.tensorboard()
plot4.tensorboard()
input("Press enter to continue...")
