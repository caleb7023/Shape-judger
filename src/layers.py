#!/user/bin/env

# author:caleb7023

from nn_calc_lib_cuda import NetworkLayer       as nl
from nn_calc_lib_cuda import ActivationFunction as ac

layers=[
    nl.Input(size=16384),

    nl.Activation(
        inputs=16384,
        neurons=1024,
        activation_function=ac.swish,
        weights_random_range=(-0.1, 0.1),
        bias_random_range=(-0.1, 0.1)
    ),

    nl.Activation(
        inputs=1024,
        neurons=512,
        activation_function=ac.swish,
        weights_random_range=(-0.1, 0.1),
        bias_random_range=(-0.1, 0.1)
    ),

    nl.Activation(
        inputs=512,
        neurons=256,
        activation_function=ac.swish,
        weights_random_range=(-0.1, 0.1),
        bias_random_range=(-0.1, 0.1)
    ),\

    nl.Activation(
        inputs=256,
        neurons=32,
        activation_function=ac.swish,
        weights_random_range=(-0.1, 0.1),
        bias_random_range=(-0.1, 0.1)
    ),

    nl.Activation(
        inputs=32,
        neurons=8,
        activation_function=ac.swish,
        weights_random_range=(-0.1, 0.1),
        bias_random_range=(-0.1, 0.1)
    ),

    nl.Activation(
        inputs=8,
        neurons=2,
        activation_function=ac.swish,
        weights_random_range=(-0.1, 0.1),
        bias_random_range=(-0.1, 0.1)
    ),

]
