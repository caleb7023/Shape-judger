#!/usr/bin/env python3

# Author: caleb7023

import numpy as np # For matrix calculation
from nn_props import props # Neural network properties
from os import mkdir # For creating directories



def create_dir(path:str)->None: # Create directory
    try:mkdir(path)             # Try to create directory
    except FileExistsError:pass # If the directory already exists, do nothing



# Create directories
create_dir("./train_data")
create_dir("./train_data/nn_props")
create_dir("./train_data/nn_props/weights")
create_dir("./train_data/nn_props/biases")


# randomize bias and weights
for i,neuron_size in enumerate(props["nn"]):

    # reset weights
    if i == 0:  # if it is the first layer                # V  The layer size                  # V  The input size
        np.save(f"./train_data/nn_props/weights/layer{i}", np.random.rand(neuron_size["size"],  props["global"]["first_input_size"]))
    else: # else
        np.save(f"./train_data/nn_props/weights/layer{i}", np.random.rand(neuron_size["size"],  props["nn"][i-1]["size"]))

    np.save(f"./train_data/nn_props/biases/layer{i}" , np.random.rand(neuron_size["size"])) # reset biases


# reset the progress data
np.save("./train_data/accuary_list", np.array([]))
with open("./train_data/terms"      , "w") as f:f.write("0")
with open("./train_data/total_fails", "w") as f:f.write("0")


# Success message
print("Reset success.")