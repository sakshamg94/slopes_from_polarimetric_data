import os
import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
import scipy.io

def mean_fields(file_location):
    
    filenames = os.listdir(file_location)
    
    for i, filename in enumerate(filenames):
        image_data = cv2.imread(os.path.join(file_location, filename).format(i))[:,:,0].astype(np.float)
        im90, im45, im135, im0 = quad_algo(image_data)
        if i==0:
            cumul90 = im90
            cumul45 = im45
            cumul135 = im135
            cumul0 = im0
        else:
            cumul90 += im90
            cumul45 += im45
            cumul135 += im135
            cumul0 += im0
    
    return cumul90/(i+1), cumul45/(i+1), cumul135/(i+1), cumul0/(i+1)

def flat_field_params(dark_files_location, flat_files_location):
    '''flat field correction is being done to correct for the optical vignetting observed in the images 
    i.e. darker edges 
    This vignetting (see wikipedia) is due to the lens not the chip itself. In multielement lens, shadows
    of the front ones fall on the edges of the rear ones. It can also be reduced by using a smaller aperture
    '''
    D90, D45, D135, D0 = mean_fields(dark_files_location)
    F90, F45, F135, F0 = mean_fields(flat_files_location)
    m90, m45, m135, m0 = np.mean(F90-D90), np.mean(F45-D45), np.mean(F135-D135), np.mean(F0-D0)
    F90D90, F45D45, F135D135, F0D0 = F90-D90, F45-D45, F135-D135, F0-D0
    return F90D90, F45D45, F135D135, F0D0,D90, D45, D135, D0, m90, m45, m135, m0


def theta_phi(filename, file_location, num_images, material, 
                  gaussian_smoothing_sigma=0, correction_angle=0, show_DOLP_curve = False,\
                  flat_field_correct = 0, flat_field_correction_params = []):
    print('Processing data for material : {}'.format(material.upper()))
        
    for i in range(num_images):
        image_data = cv2.imread(os.path.join(file_location, filename).format(i))[:,:,0]
        print("image_data shape is : ", image_data.shape)
        #     print(not (image_data[:,:,0]==image_data[:,:,2]).all())
        im90, im45, im135, im0 = quad_algo(image_data)
        print("im90 shape is : ", im90.shape)
        
        #   write_quad_image(im90, 90, file_location, filename[:-5])
        if flat_field_correct != 0:
            print("making flat field corrections..................")
            F90D90, F45D45, F135D135, F0D0,D90, D45, D135, D0, m90, m45, m135, m0 = flat_field_correction_params
            im90 = (im90 - D90)/F90D90*m90
            im45 = (im45 - D45)/F45D45*m45
            im135 = (im135 - D135)/F135D135*m135
            im0 = (im0 - D0)/F0D0*m0
        
        #gaussian filtering
        # 3.64px/mm or 0.27mm/px ; 2.7mm capillary length scale => 3.64*2.7 = 9.83px (~10px); 
        # 5px on each side to get x-y length scale of ~10px. 5px = 2sigma =>sigma = 2.5 both in x and y. Don't worry about the diagonal -- it has the least weight anyway in the convolution filter. kernel size is the coefficient of the gaussian [here (see examplke 40-1)](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-40-incremental-computation-gaussian)

        # See here for how to use the gasussian kernel: 
        # https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
        if gaussian_smoothing_sigma!=0: # shoudl be 2.5 for water
            print("gaussian smoothing.............................")
            im0 = cv2.GaussianBlur(im0,ksize = (0,0), sigmaX =gaussian_smoothing_sigma)
            im90 = cv2.GaussianBlur(im90,ksize = (0,0), sigmaX =gaussian_smoothing_sigma)
            im45 = cv2.GaussianBlur(im45,ksize = (0,0), sigmaX =gaussian_smoothing_sigma)
            im135 = cv2.GaussianBlur(im135,ksize = (0,0), sigmaX =gaussian_smoothing_sigma)
        
        # Get Stokes vector first 3 components from 4 quad images
        # S is floating point array -- cannot be written as a tiff or png "image"
        im_stokes0, im_stokes1, im_stokes2 = calculate_Stokes_Params(im90, im45, im135, im0, correction_angle)
        print("im_stokes0 shape is : ", im_stokes0.shape) 
        
        # Get DOLP from Stokes measurements
        im_DOLP = calculate_DOLP(im_stokes0, im_stokes1, im_stokes2)
        print("im_DOLP shape is : ", im_DOLP.shape)
        
        # Get the material refractive index for AOI or DOLP calculations
        if material.lower() == 'borosilicate_glass':
            nt = 1.47
        elif material.lower() == 'ferrofluid_emg905':
            nt = 1.58
        elif material.lower() == 'water':
            nt = 1.33
        else:
            raise Exception('Check the material type entered: borosilicate_glass or ferrofluid_EMG905 or water')
        
        # Get theta (Angle of incidence) from DOLP
        im_theta, DOLP_errors =  DOLP2Theta(1, nt, im_DOLP, show_DOLP_curve)
        print(np.array(DOLP_errors).any())

        # Create a heatmap from the greyscale image
        # floating point array -- cannot be written as a tiff or png "image", write a matlab array instead
#         im_DOLP_normalized = normalise_DOLP(im_DOLP)    
#         im_DOLP_heatmap = heatmap_from_greyscale(im_DOLP_normalized)
#         write_float_image("DOLP", i, im_DOLP, file_location, filename[:-5])

        # Get AOLP from Stokes measurements
        im_AOLP = calculate_AOLP(im_stokes1, im_stokes2)

        # Get angle of plane of polarization from AOLP
        im_phi = (im_AOLP + np.pi/2)*180/np.pi;

        # Create a heatmap from the greyscale image
        # floating point array -- cannot be written as a tiff or png "image", write a matlab array instead
#         im_AOLP_normalized = normalise_AOLP(im_AOLP)    
#         im_AOLP_heatmap = heatmap_from_greyscale(im_AOLP_normalized)
#         write_float_image("AOLP", i, im_AOLP, file_location, filename[:-5])    
        
        return im_theta, im_phi


def quad_algo(image_data):
    '''
    Extract image data from each of the 4 polarized quadrants
    '''
    tic = time.time()

    im90 = image_data[0::2, 0::2]
    im45 = image_data[0::2, 1::2]
    im135 = image_data[1::2, 0::2]
    im0 = image_data[1::2, 1::2]

    toc = time.time()
    # print("Quad algorithm time ", toc - tic, "s")
    
    return im90, im45, im135, im0

def write_quad_image(quad_image, quad_angle, location, file_prefix):
    cv2.imwrite(os.path.join(location, file_prefix+"_"+"IM{}.tiff".format(quad_angle)), quad_image)

def write_glare_reduced_image(im90, im45, im135, im0, location):
    '''
    Calculate a glare reduced image by taking the lowest pixel value from each quadrant
    '''
    tic = time.time()

    glare_reduced_image = np.minimum.reduce([im90, im45, im135, im0])

    toc = time.time()
    print("Glare reduction time: ", toc - tic, "s")

    cv2.imwrite(os.path.join(location, file_prefix+"_"+"GlareReduced-{}.tiff".format(i)), glare_reduced_image)
    
    return glare_reduced_image 

def calculate_Stokes_Params(im90, im45, im135, im0, correction_angle = 0):
    '''
    Calculate Stokes parameters
    correction_angle (degrees): non-zero angle to convert from lab frame to meridional frame measurements
    '''
    tic = time.time()

    im_stokes0 = im0.astype(np.float) + im90.astype(np.float) #lab_frame measurement
    im_stokes1 = im0.astype(np.float) - im90.astype(np.float) #lab_frame measurement
    im_stokes2 = im45.astype(np.float) - im135.astype(np.float)#lab_frame measurement
    
    if correction_angle!=0: # convert to meridional frame
        print("correcting for a correction_angle..................")
        im_stokes2 = -im_stokes1*np.sin(2*np.deg2rad(correction_angle))+im_stokes2*np.cos(2*np.deg2rad(correction_angle))
        im_stokes1 = im_stokes1*np.cos(2*np.deg2rad(correction_angle))+ im_stokes2*np.sin(2*np.deg2rad(correction_angle))
    
    toc = time.time()
    print("Stokes time: ", toc - tic, "s")
    
    return im_stokes0, im_stokes1, im_stokes2

    
def calculate_DOLP(im_stokes0, im_stokes1, im_stokes2):
    '''
    Calculate DoLP
    '''
    tic = time.time()

    im_DOLP = np.divide(
        np.sqrt(np.square(im_stokes1) + np.square(im_stokes2)),
        im_stokes0,
        out=np.zeros_like(im_stokes0),
        where=im_stokes0 != 0.0,
    ).astype(np.float)

    im_DOLP = np.clip(im_DOLP, 0.0, 1.0)
    toc = time.time()
    print("DoLP time: ", toc - tic, "s")
    
    return im_DOLP

def normalise_DOLP(im_DOLP):
    '''
    Normalize from [0.0, 1.0] range to [0, 255] range (8 bit)
    '''
    im_DOLP_normalized = (im_DOLP * 255).astype(np.uint8)
    return im_DOLP_normalized

def DOLP_curve(ni, nt, show_DOLP_curve):
    '''
    Get the theoretical DOLP curve for unpolarized light transmission 
    from incident medium with refractive index ni to the transmission medium
    of refractive index nt
    '''
    #nt: 1.47 borosilicate glass; %1.58 ferrofluid emg 905);%1.333 water; 57.68
    
    th_Brewster = np.arctan2(nt,ni)
    
    thI = np.linspace(0, th_Brewster, 200).astype(
        np.float
    )
#     thI = np.linspace(th_Brewster, np.pi/2,200).astype(
#         np.float
#     )
    thT = np.arcsin(ni*np.sin(thI)/nt).astype(
        np.float
    )

    alpha = 0.5*(np.tan(thI-thT)/np.tan(thI+thT))**2
    eta = 0.5*(np.sin(thI-thT)/np.sin(thI+thT))**2

    DOLP = np.abs((alpha - eta)/(alpha + eta))
    
    if show_DOLP_curve:
        fig = plt.figure(figsize=(5, 5),  facecolor='w', edgecolor='k')
        plt.plot(thI*180/np.pi, np.abs(DOLP),color='Indigo', linestyle='--', linewidth=3)
        plt.axvline(x=th_Brewster*180/np.pi, linestyle='-', linewidth=3)
        plt.xlabel("Incidence angle ($\\theta_i$) [deg]", fontsize=16)
        plt.ylabel("DOLP($\\theta_i$)", fontsize = 16)
        plt.xlim(0,90)
        plt.grid()
        plt.show()

    return DOLP, thI

def DOLP2Theta(ni, nt, im_DOLP, show_DOLP_curve):
    '''
    Convert from measured DOLP to theta using theoretical interpolation
    '''
    theoretical_DOLP_curve, theta_in_radians = DOLP_curve(ni, nt, show_DOLP_curve)

    f = interpolate.interp1d(theoretical_DOLP_curve, theta_in_radians*180/np.pi)
    im_theta = np.zeros_like(im_DOLP)
    
    DOLP_errors = []
    tic = time.time()
    for i in range(im_theta.shape[0]):
        for j in range(im_theta.shape[1]):
            try:
                im_theta[i,j] = f(im_DOLP[i,j])
            except ValueError:
                DOLP_errors.append(im_DOLP[i,j])
    
    toc = time.time()
    print("Time for DOLP2Theta conversion is : ", toc-tic, "seconds")
    return im_theta, DOLP_errors

def write_float_image(im_prefix, im_number, image, location, file_prefix):
    fname = os.path.join(location, file_prefix+"_"+"{}-{}.mat".format(im_prefix, im_number))
    scipy.io.savemat(fname, {str(im_prefix): image})

def heatmap_from_greyscale(grey_image):
    '''
    @ param: grey_image : [0, 255] range (8 bit) image array
    '''
    # Create a heatmap from the greyscale image
    im_heatmap = cv2.applyColorMap(
        grey_image, cv2.COLORMAP_JET
    )
    return im_heatmap

def calculate_AOLP(im_stokes1, im_stokes2):
    '''
    Calculate AoLP
    '''
    tic = time.time()

    inverted_angle = np.arctan2(im_stokes2,im_stokes1) # lies in [-pi,pi] -- 
    # it's teh 4 quadrant inverse    
    
    im_AOLP = (0.5 * inverted_angle).astype(np.float)
    # ultimately phi is above im_AOLP + np.pi/2 --> this manipulation above (x0.5) 
    # and addition of np.pi/2 ensure that 
    # the resulting phi lies in [0,pi] as is expected physically 

    toc = time.time()
    print("AoLP time: ", toc - tic, "s")
    return im_AOLP

def normalise_AOLP(im_AOLP):
    '''
    Normalize from [-pi/2, pi/2] range to [0, 255] range (8 bit)
    '''
    im_AOLP_normalized = (
        (im_AOLP + math.pi / 2) * (255 / math.pi)
    ).astype(np.uint8)
    return im_AOLP_normalized