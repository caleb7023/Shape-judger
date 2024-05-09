#!/usr/bin/env python3

# Author: caleb7023

import numpy as np

from random import randrange, getrandbits


# Generate a array of X and Y pos of 128x128 array
Size = np.arange(0, 128)
pos_x_array, pos_y_array = np.meshgrid(Size, Size)
pos_x_array = np.array(np.float64(pos_x_array))
pos_y_array = np.array(np.float64(pos_y_array))
del Size

__all__ = [
    "swish"                  ,
    "filter"                 ,
    "softmax"                ,
    "create_ellipse_128x128" ,
    "create_random_shape_img",
]

class act_func: # Activation functions

    # \text{Swish}(x) = x \cdot \sigma(x)
    def swish(input:float)->float:
        return input*act_func.sigmoid(input)

    # \sigma(x) = \frac{1}{1 + e^{-x}}
    def sigmoid(input:float)->float:
        return 1.0 / (1.0+np.exp(-input))

    # \cost(x, y) = (x-y)^2
    def cost(input:float, target:float)->float:
        return 1.0 / (1.0+np.exp(-input))
    
    class grad:

        def swish(input:float)->float:
            sigmoid_memo = act_func.sigmoid(input)
            return act_func.swish(input)*(1-sigmoid_memo) + sigmoid_memo

        def sigmoid(input:float)->float:
            return act_func.sigmoid(input)*(1-act_func.sigmoid(input))

        def cost(input:float, target:float)->float:
            return 2*(input-target)



# Render ellipse.
# The Pos1 should be smaller than Pos2.
def create_ellipse_128x128(pos_1:tuple, pos_2:tuple) -> np.array:

    global pos_x_array, pos_y_array

    # Calc the center of the ellipse
    ellipse_center = ((pos_1[0] + pos_2[0]) * 0.5, (pos_1[1] + pos_2[1]) * 0.5)

    # Width ratio
    width_ratio = ((pos_2[0] - pos_1[0]) / (pos_2[1] - pos_1[1]))

    # The radius of the x axis
    x_radius = (pos_2[0] - pos_1[0]) * 0.5

    temp_pos_x_array, temp_pos_y_array = np.array(pos_x_array), np.array(pos_y_array)

    temp_pos_x_array -= ellipse_center[0]
    temp_pos_y_array -= ellipse_center[1]

    # To create the ellipse as circle
    temp_pos_y_array *= width_ratio

    # Calc all the distance from Pos1
    distance_array = np.sqrt(np.square(temp_pos_x_array) + np.square(temp_pos_y_array))

    # If the distance were smaller than the X radius, the aug gonna be True
    return distance_array < x_radius



def create_random_shape_img() -> np.array:

    # Pos1
    pos_1 = (randrange(0, 100), randrange(0, 100))

    # Pos2, will be bigger than Pos1 value
    pos_2 = (randrange(pos_1[0] + 27, 127), randrange(pos_1[1] + 27, 127))

    # if getrandbits(1) returns 1, its gonna create img of a rectangle.
    # else its gonna return img of a ellipse.
    is_rectangle = getrandbits(1)

    if is_rectangle:

        # Create a img with bool
        Img = np.zeros((128 , 128) , bool)

        # Get an img of a rectangle
        Img[pos_1[0] : pos_2[0],
            pos_1[1] : pos_2[1]] = True
    else:
        # Get an img of an ellipse
        Img = create_ellipse_128x128(pos_1, pos_2)

    for i in range(randrange(0, 3)):
        Img = np.rot90(Img)

    return Img, is_rectangle



def filter(img, kernel):
    img
    return kernel