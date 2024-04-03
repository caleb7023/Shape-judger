#!/usr/bin/env python3

# Author: caleb7023

import numpy as np

np.save("./train_data/neuron_weights/neuron_weights_ellipse"  , np.random.rand(128, 128))
np.save("./train_data/neuron_weights/neuron_weights_rectangle", np.random.rand(128, 128))

np.save("./train_data/accuary_list", np.array([]))

with open("./train_data/terms"      , "w") as f:f.write("0")
with open("./train_data/total_fails", "w") as f:f.write("0")

print("Reset success.")