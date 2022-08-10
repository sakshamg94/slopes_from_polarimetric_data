import numpy as np
import os
import time
import pickle
from tqdm import tqdm
import argparse
from scipy.io import savemat
from klepto.archives import dir_archive

from angle2slope import *

################
# SET UP ARG PARSER OBJECT
parser = argparse.ArgumentParser()
parser.add_argument("bed")
parser.add_argument("flow")
parser.add_argument("submergence")
parser.add_argument("bed_data_2022_04") # flag to ensure latest 1hr long dataset is used 

# READ ARGUMENTS
args = parser.parse_args()

# USE THE ARGUMENTS AS YOU WISH
BEDFORM = str(args.bed)
FLOW_SPEED = str(args.flow)
SUBMERGENCE = str(args.submergence)
bed_data_2022_04 = bool(int(args.bed_data_2022_04))
TEST = 3

### CORE

data_source = "../raw2angle_pickles_modifAOLP_longerExpt/"
save_location = "../angle2slope_mat_files_modifThetaPhi_longerExpt/"
for SUBFOLDER in tqdm(range(1, 49)):  # range(1,5) if older experiment data
    # SPECIFY LOCATION

    data_foldername = f"{BEDFORM}_{FLOW_SPEED}Flow_{SUBMERGENCE}H_test{TEST}_subFolder{SUBFOLDER}"
    save_dest = save_location + data_foldername + '.mat'

    # LOAD DATA
    data_path = os.path.join(data_source, data_foldername)
    assert os.path.exists(data_path) == True
    tic = time.time()
    data = dir_archive(data_path, {}, serialized=True, cached=True)

    # process the data to get the slope
    print("bed data 2022_04 theta phi processing...")
    data.load("all_theta_maps", "all_phi_maps")
    print(f"time taken to load data = {time.time() - tic}s)")
    theta = np.transpose(data['all_theta_maps'],
                         (2, 0, 1)).astype(np.float32)
    phi = np.transpose(data['all_phi_maps'], (2, 0, 1)).astype(np.float32)
    #print("data for  bed_data_2022_04", theta, phi)
    print(theta.shape, phi.shape)


    # GET THE SLOPES
    along_look_slope, cross_look_slope = angle2slope(
        theta, phi, mean_subtraction=True)
    # save the slope maps as raw data for later processing
    dic = {"BEDFORM": BEDFORM, "FLOW_SPEED": FLOW_SPEED, "SUBMERGENCE": SUBMERGENCE, "TEST": TEST,
           "SUBFOLDER": SUBFOLDER, "along_look_slope": along_look_slope, "cross_look_slope": cross_look_slope}
    savemat(save_dest, dic)
    print("DONE!")
    data = dic = theta = phi = dtheta = dphi = phi_ref = theta_ref = along_look_slope = cross_look_slope = None
