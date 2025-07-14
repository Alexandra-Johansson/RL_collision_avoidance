import matplotlib.pyplot as plt
import numpy as np
from data_handling import Plot

model_filepath = "Training/results/PPO_training_2025.07.04-23.29.23"
model_filepath2 = "Training/results/PPO_training_2025.07.05-12.50.08"
model_filepath3 = "Training/results/PPO_training_2025.07.06-01.54.55"
model_filepath4 = "Training/results/PPO_training_2025.07.06-04.32.42"

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
