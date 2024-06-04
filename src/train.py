#!/user/bin/env

# author:caleb7023

import nn_calc_lib_cuda as nclc

import cupy as cp

from layers import layers

from random import randrange, getrandbits

# Render ellipse.
# The Pos1 should be smaller than Pos2.
def create_ellipse_128x128(pos_1:tuple, pos_2:tuple) -> cp.ndarray:
    # Calculate the center of the ellipse
    center = ((pos_1[0] + pos_2[0]) // 2, (pos_1[1] + pos_2[1]) // 2)
    # Calculate the radii of the ellipse
    radius_x = abs(pos_2[0] - pos_1[0]) // 2
    radius_y = abs(pos_2[1] - pos_1[1]) // 2
    # Create a grid of coordinates
    x, y = cp.meshgrid(cp.arange(128), cp.arange(128))
    # Calculate the distance from each point to the center of the ellipse
    distance = ((x - center[0]) / radius_x) ** 2 + ((y - center[1]) / radius_y) ** 2
    # Create the ellipse image
    ellipse_img = distance <= 1
    return ellipse_img.astype(bool)

def create_img(is_rectangle) -> cp.ndarray:
    # Pos1
    pos_1 = (randrange(0, 100), randrange(0, 100))
    # Pos2, will be bigger than Pos1 value
    pos_2 = (randrange(pos_1[0] + 27, 127), randrange(pos_1[1] + 27, 127))
    if is_rectangle:
        # Create a img with bool
        Img = cp.zeros((128 , 128) , bool)
        # Get an img of a rectangle
        Img[pos_1[0] : pos_2[0],
            pos_1[1] : pos_2[1]] = True
    else:
        # Get an img of an ellipse
        Img = create_ellipse_128x128(pos_1, pos_2)
    for i in range(randrange(0, 3)):
        Img = cp.rot90(Img)
    return Img


def main()->None:
    # create the neural network
    neural_network = nclc.Network(layers=layers)
    total_fails = 0
    total_terms = 0
    EACH_TERMS = 1000
    is_rectangle = True
    while True:
        fails = 0
        for i in range(EACH_TERMS):
            # Create a random shape img
            is_rectangle = not is_rectangle
            img = create_img(is_rectangle)
            # Flatten the img
            img = img.flatten()
            # Forward propagation
            neural_network.forward(img)
            # Get the result
            result = neural_network.value
            # Increase the fails if the result was wrong
            if (result[1] < result[0]) != is_rectangle or result[0] == result[1]:
                fails += 1
            # Backward propagation
            target_value = cp.array([is_rectangle, not is_rectangle])
            neural_network.backward(target_value, 0.01)
            if not is_rectangle:
                neural_network.update()
            if False: # debugging
                print(f"{neural_network.value}, \n{target_value.astype(float)}\n")
        total_terms += EACH_TERMS
        total_fails += fails
        print(f"fails:{fails}, total_fails:{total_fails}, accuracy:{round(1-fails/EACH_TERMS, 4)}, total_accuracy:{round(1-total_fails/total_terms, 4)}, total_terms:{total_terms}")

if __name__ == "__main__":main()