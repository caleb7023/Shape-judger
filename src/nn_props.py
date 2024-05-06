#!/usr/bin/env python3

# Author: caleb7023


# All activation functions are set to swish.
# The size at here means the size of the output of the layer.
props = {
    "global": {
        "first_input_size": 16384, # 128x128
    },
    "nn": [ # Neural network
        {
            "size": 512 # 16384 (128x128) input
        },              #  |
        {               #  |
            "size": 384 # 512 input
        },              #  |
        {               #  |
            "size": 256 # 384 input
        },              #  |
        {               #  |
            "size": 128 # 256 input
        },              #  |
        {               #  |
            "size": 64  # 128 input
        },              #  |
        {               #  |
            "size": 2   # 64 input
        }
    ]
}