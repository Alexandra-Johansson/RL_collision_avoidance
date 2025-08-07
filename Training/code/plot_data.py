from data_handling import Plot

model_filepath = "Training/results/PPO_training_2025.08.07-01.47.19"
model_filepath2 = "Training/results/PPO_training_2025.08.07-04.53.13"
model_filepath3 = "Training/results/PPO_training_2025.08.07-07.35.27"
model_filepath4 = "Training/results/PPO_training_2025.08.07-23.03.40"

plot = Plot(model_filepath)
plot2 = Plot(model_filepath2)
plot3 = Plot(model_filepath3)
plot4 = Plot(model_filepath4)
#plot.plotReward()
plot.tensorboard()
plot2.tensorboard()
plot3.tensorboard()
plot4.tensorboard()
input("Press enter to continue...")
