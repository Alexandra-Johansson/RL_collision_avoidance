import numpy as np
import matplotlib.pyplot as plt

from data_handling import Plot

model_filepath = "Training/results/PPO_training_24.06.2025-10.44.29"

plot = Plot(model_filepath)
plot.plotReward()
plot.tensorboard()
input("Press enter to continue...")
