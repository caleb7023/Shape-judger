#!/usr/bin/env python3

# Author: caleb7023

import numpy as np # For matrix calculation
import time # For measuring calculation time
import func # Functions for rendering and activation functions
import os # 
from nn_props import props # Neural network properties


def train(Img, learning_rate:float, is_rectrangle:bool)->bool:

    global neuron_weights_ellipse, neuron_weights_rectangle,\
           ellipse_bias          , rectangle_bias

    output = Img.flatten()

    for prop in props["nn"]:

        output = func.act_func.swish(np.dot(output, np.random.rand(prop["size"], len(output))))

    return False



def load_datas():
    
    global neuron_weights_ellipse, neuron_weights_rectangle,\
           ellipse_bias          , rectangle_bias          ,\
           terms                 , total_fails             ,\
           accuary_list

    nn=[]
    
    for i in range(len(os.listdir("./train_data/nn_props/weights"))):
        nn += [{
            "weights":np.load(f"./train_data/nn_props/weights/layer{i[0]}.npy"),
            "biases" :np.load(f"./train_data/nn_props/biases/layer{i[0]}.npy" )
        }]

    ellipse_bias             = np.load("./train_data/biases.npy")[0]
    rectangle_bias           = np.load("./train_data/biases.npy")[1]
    
    terms       = int(open("./train_data/terms"      , "r").read())
    total_fails = int(open("./train_data/total_fails", "r").read())
    
    accuary_list = np.load("./train_data/accuary_list.npy")



def main(save_to_disk:bool = True, learning_rate:float=0.001):

    global neuron_weights_ellipse, neuron_weights_rectangle,\
           ellipse_bias          , rectangle_bias

    load_datas()

    while True:

        start_time = time.time()

        fails = 0
        
        for i in range(50000):

            terms += 1

            img, is_rectrangle = func.create_random_shape_img()

            fails += train(img, 0.001, is_rectrangle)
            

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