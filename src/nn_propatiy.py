#!/user/bin/env

# author:caleb7023

from nn_calc_lib import ActivationFunction

prop=[
    {
        "neuron_size": 16384,
        "is_input_layer": True
    },
    {
        "neuron_size": 256,
        "activation_function": ActivationFunction.swish,
        "derivative_function": ActivationFunction.derivative.swish
    },
    {
        "neuron_size": 128,
        "activation_function": ActivationFunction.swish,
        "derivative_function": ActivationFunction.derivative.swish
    },
    {
        "neuron_size": 32,
        "activation_function": ActivationFunction.swish,
        "derivative_function": ActivationFunction.derivative.swish
    },
    {
        "neuron_size": 8,
        "activation_function": ActivationFunction.swish,
        "derivative_function": ActivationFunction.derivative.swish
    },
    {
        "neuron_size": 2,
        "activation_function": ActivationFunction.sigmoid,
        "derivative_function": ActivationFunction.derivative.sigmoid
    },
]