#!/user/bin/env

# author:caleb7023

import os
import numpy as np
import shutil
import json as js



def reset_status()->None:

    os.mkdir("data/status") # create the folder to save the statistics
    # reset the statistics
    with open("data/status/status.json", "a") as f:
        js.dump({"terms":0, "fails":0}, f)



def main()->None:
    # reset the data folder
    shutil.rmtree("data") # remove the data folder
    os.mkdir("data") # create the folder to save the datas
    # reset the dataset
    reset_status() # reset the status



if __name__ == "__main__":main()
























































































#                                                                                                                                                            
#                  ■■                                                               ■■                   ■■                                ■      ■          
#           ■■■■   ■■                                                               ■                     ■                                ■     ■■          
#          ■■  ■                                                                    ■                                                      ■     ■■          
#          ■       ■■   ■■■■■ ■■■■■■  ■■■■        ■■■■■■  ■■■   ■■ ■■■  ■■■         ■■■■   ■ ■■■  ■■■■   ■■   ■■■■■          ■■■■   ■■■    ■     ■■    ■■■■  
#          ■       ■■   ■ ■ ■  ■ ■ ■  ■  ■        ■■ ■ ■  ■  ■   ■■    ■  ■■        ■■ ■■  ■■■    ■  ■    ■    ■■ ■■        ■  ■   ■  ■■   ■     ■■    ■  ■  
#          ■       ■■   ■ ■ ■■ ■ ■ ■ ■■  ■■       ■■ ■ ■ ■■  ■   ■     ■   ■        ■   ■  ■■        ■    ■    ■  ■■        ■      ■   ■   ■     ■■   ■■     
#          ■ ■■■   ■■   ■ ■ ■■ ■ ■ ■ ■■■■■■       ■■ ■ ■ ■   ■■  ■    ■■■■■■        ■   ■  ■       ■■■    ■    ■  ■■        ■■■   ■■■■■■   ■     ■■    ■■■   
#          ■   ■   ■■   ■ ■ ■■ ■ ■ ■ ■            ■■ ■ ■ ■   ■■  ■    ■■            ■   ■  ■■     ■  ■    ■    ■  ■■          ■■■ ■■       ■     ■■      ■■  
#          ■   ■   ■■   ■ ■ ■■ ■ ■ ■ ■■           ■■ ■ ■ ■■  ■   ■     ■            ■   ■  ■     ■■  ■    ■    ■  ■■            ■  ■       ■     ■■       ■  
#          ■■  ■   ■■   ■ ■ ■■ ■ ■ ■  ■           ■■ ■ ■  ■  ■   ■     ■■           ■■ ■■  ■■    ■■  ■    ■    ■  ■■        ■  ■■  ■■      ■      ■   ■■  ■  
#           ■■■■   ■■   ■ ■ ■■■■ ■ ■■  ■■■        ■■ ■ ■  ■■■   ■■      ■■■■        ■■■■   ■■     ■■■■   ■■   ■■  ■■        ■■■■    ■■■■   ■■■    ■■■  ■■■■  
#                                                                                                                                                          