#!/usr/bin/env python3

# Author: caleb7023

import numpy as np

import time

import func



def train(Img, is_rectrangle:bool)->bool:

    global neuron_weights_ellipse, neuron_weights_rectangle,\
           ellipse_bias          , rectangle_bias

    ellipse_score    = np.sum(neuron_weights_ellipse   * Img) + ellipse_bias
    rectrangle_score = np.sum(neuron_weights_rectangle * Img) + rectangle_bias

    softmax_output = func.softmax(np.array([ellipse_score, rectrangle_score]))

    judged_shape_is_rectrangle = softmax_output[0] < softmax_output[1]

    if judged_shape_is_rectrangle != is_rectrangle:
        if judged_shape_is_rectrangle:
            neuron_weights_ellipse   += Img * (softmax_output[0]-1) ** 2
            neuron_weights_rectangle -= Img *  softmax_output[1]    ** 2
            ellipse_bias   += 1
            rectangle_bias -= 1
        else:
            neuron_weights_ellipse   -= Img *  softmax_output[0]    ** 2
            neuron_weights_rectangle += Img * (softmax_output[1]-1) ** 2
            ellipse_bias   -= 1
            rectangle_bias += 1
        return True

    return False



def main(save_to_disk:bool = True):

    global neuron_weights_ellipse, neuron_weights_rectangle,\
           ellipse_bias          , rectangle_bias

    neuron_weights_ellipse   = np.load("./train_data/neuron_weights/neuron_weights_ellipse.npy"  )
    neuron_weights_rectangle = np.load("./train_data/neuron_weights/neuron_weights_rectangle.npy")

    biases = np.load("./train_data/biases.npy")
    ellipse_bias   = biases[0]
    rectangle_bias = biases[1]

    accuary_list = np.load("./train_data/accuary_list.npy")
    
    with open("./train_data/terms"      , "r") as f: terms       = int(f.read())
    with open("./train_data/total_fails", "r") as f: total_fails = int(f.read())

    while True:

        start_time = time.time()

        fails = 0
        
        for i in range(50000):

            terms += 1

            img, is_rectrangle = func.create_random_shape_img()

            fails += train(img, is_rectrangle)
            

        total_fails += fails

        accuary_list = np.append(accuary_list, fails * 0.00002)

        ##########################
        # Save datas to the disk #
        ##########################
        
        if save_to_disk:

            np.save("./train_data/neuron_weights/neuron_weights_ellipse"  , neuron_weights_ellipse  )
            np.save("./train_data/neuron_weights/neuron_weights_rectangle", neuron_weights_rectangle)

            np.save("./train_data/biases", np.array([ellipse_bias, rectangle_bias]))
            
            np.save("./train_data/accuary_list", accuary_list)

            with open("./train_data/terms"      , "w") as f: f.write(str(terms))
            with open("./train_data/total_fails", "w") as f: f.write(str(total_fails))

        ###############
        # Print infos #
        ###############

        print("Terms:{0}, Total_fails:{1}, Acuracy:{2}%, Sec_per_10000_time:{3}".format(terms,
                                                                                        total_fails,
                                                                                        round(fails * 0.002, 1),
                                                                                        round((time.time() - start_time) * 0.2, 3)))



if __name__ == "__main__":
    main(save_to_disk=True)


















































































# Im out of brain cells rn