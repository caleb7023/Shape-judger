#!/user/bin/env

# author:caleb7023

import nn_calc_lib as ncl

import numpy as np

from nn_propatiy import prop

from random import randrange, getrandbits

# Render ellipse.
# The Pos1 should be smaller than Pos2.
def create_ellipse_128x128(pos_1:tuple, pos_2:tuple) -> np.array:
    # Calculate the center of the ellipse
    center = ((pos_1[0] + pos_2[0]) // 2, (pos_1[1] + pos_2[1]) // 2)
    # Calculate the radii of the ellipse
    radius_x = abs(pos_2[0] - pos_1[0]) // 2
    radius_y = abs(pos_2[1] - pos_1[1]) // 2
    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(128), np.arange(128))
    # Calculate the distance from each point to the center of the ellipse
    distance = ((x - center[0]) / radius_x) ** 2 + ((y - center[1]) / radius_y) ** 2
    # Create the ellipse image
    ellipse_img = distance <= 1
    return ellipse_img.astype(bool)

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


def main()->None:
    # create the neural network
    neural_network = ncl.NeuralNetwork(propatiy=prop)
    total_fails = 0
    total_terms = 0
    while True:
        fails = 0
        for i in range(100):
            # Create a random shape img
            img, is_rectangle = create_random_shape_img()
            # Flatten the img
            img = img.flatten()
            # Forward propagation
            neural_network.forward_propagation(img)
            # Get the result
            result = neural_network.value
            # Increase the fails if the result was wrong
            if (result[1] < result[0]) != is_rectangle:
                fails += 1
            # Backward propagation
            neural_network.backward_propagation(np.array([is_rectangle, not is_rectangle]))
        total_terms += 100
        total_fails += fails
        print(f"fails:{fails}, total_fails:{total_fails}, accuracy:{1-fails/100}, total_accuracy:{1-total_fails/total_terms}")

if __name__ == "__main__":main()