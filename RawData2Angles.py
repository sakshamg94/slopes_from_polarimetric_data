import numpy as np
from tqdm import tqdm as tqdm
import argparse
import time
from klepto.archives import dir_archive
from natsort import os_sorted
import cv2

from RawData2Angles_FLIRhelper import *

################
# SET UP ARG PARSER OBJECT
parser = argparse.ArgumentParser()
parser.add_argument("bed")
parser.add_argument("flow")
parser.add_argument("submergence")
parser.add_argument("k_val")

# READ ARGUMENTS
args = parser.parse_args()

# USE THE ARGUMENTS AS YOU WISH
BEDFORM = str(args.bed)
FLOW_SPEED = str(args.flow)
SUBMERGENCE = str(args.submergence)
k_val = int(args.k_val)  # 0,1,2,3 ; k_val = SUBFOLDER-1

##########################
ROWS_RAW = 1024 # resolution
COLS_RAW = 1224
NUM_IMAGES = 1
TEST = 3

# source data folders
# for the bedform flow case, uncomment the line below
file_location = "../bed_data_2022_04_20/{}_{}_{}_test{}_alpha35".format(BEDFORM, FLOW_SPEED,
                                                                        SUBMERGENCE, TEST)


save_location = "../raw2angle_pickles_modifAOLP_longerExpt"

print(file_location, os.path.exists(file_location))

##############################
filenames = os.listdir(file_location)
print(len(filenames))

filenames = os.listdir(file_location)
filenames = [f for f in filenames if f.endswith('.tiff')]
# [5400:10800] for second half of the data, [:5400] for first
filenames = os_sorted(filenames)[:10800]
NUM_FRAMES = len(filenames)
print("number of frames = ", NUM_FRAMES)
assert NUM_FRAMES == 5400*2  # *2 is because you are processing full data
# print for sanity
for f in filenames[:20]:
    print(f)

#################################################
for k in [k_val]:
    start_time = time.time()
    # set the frames to be processed and save locations
    SUBFOLDER = k+1

    # for the bedform case uncomment the line below
    data_foldername = "{}_{}Flow_{}H_test{}_subFolder{}".format(BEDFORM, FLOW_SPEED,
                                                                SUBMERGENCE, TEST, SUBFOLDER)
    save_destination = os.path.join(save_location, data_foldername)
    print(save_destination)

    # process the full data worth 1 hr at once using the next 4 lines of code
    if k != (NUM_FRAMES/225-1):  # remove the  +24 for the first half of the d$
        files = filenames[(k)*NUM_FRAMES//48: ((k)+1)*NUM_FRAMES//48]
    else:
        files = filenames[(k)*NUM_FRAMES//48:]

    print("files are: ", files)
    all_theta_maps = np.zeros((ROWS_RAW, COLS_RAW, len(files)))
    all_phi_maps = np.zeros((ROWS_RAW, COLS_RAW, len(files)))

    i = -1
    for filename2 in tqdm(files):
        i += 1
        im_theta2, im_phi2 = theta_phi(filename2, file_location,
                                       material='water',
                                       flat_field_correct=0,
                                       gaussian_smoothing_sigma=0,
                                       num_images=NUM_IMAGES, correction_angle=0,
                                       flat_field_correction_params=[])
        all_theta_maps[:, :, i] = np.radians(im_theta2)
        all_phi_maps[:, :, i] = np.radians(im_phi2)

    # Save using Klepto -- not pickle : directly to disk -- not memory intensive
    tic = time.time()
    demo = dir_archive(save_destination, {}, serialized=True, cached=False)

    demo['all_theta_maps'] = all_theta_maps
    demo['all_phi_maps'] = all_phi_maps

    toc = time.time()
    print('time taken = ', toc - tic)
    demo.dump()

    # Empty some space
    files = filenames = demo = None
    print(f"k = {k} took {(time.time() - start_time)/60/60}hrs.")
