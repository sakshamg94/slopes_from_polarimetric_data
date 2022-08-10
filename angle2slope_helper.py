import numpy as np

def get_theta(dtheta, theta_ref):
    '''
    Function not needed with the latest version of the code 
    where you only save the theta, not dtheta
    Author: Saksham Gakhar
    Last edited: 04/22/2022
    '''
    # get absolute values of theta
    theta = dtheta +  theta_ref # undo the reference subtraction due to raw data
    return theta

def get_phi(dphi, phi_ref):
    '''
    Function not needed with the latest version of the code 
    where you only save the phi, not dphi
    Author: Saksham Gakhar
    Last edited: 04/22/2022
    '''
    # get absolute values of phi
    phi = dphi +  phi_ref # undo the reference subtraction due to raw data
    return phi

def angle2slope(theta, phi, mean_subtraction = True):
    '''
    dtheta  = theta_measured - theta_ref in radians
    dphi = phi_measured - phi_ref in radians
    theta is the zenith angle or the incidence angle: lies in 0 to pi/2 radians range
    phi is the azimuth angle or the phase angle or the polarization angle. 
    Lies in 0 to 2*pi radians
    Use absolute theta as phi for slope calculations

    Author: Saksham Gakhar
    Last edited: 04/13/2022
    '''

    # equations for the slope 
    # (see Advances in Photometric 3D-Reconstruction by Durou et al. Springer book/ Nathan email)
    along_look_slope = np.sin(theta) * np.sin(phi) # streamwise
    cross_look_slope = np.sin(theta) * np.cos(phi) # spanwise

    # get rid of the mean slope for each frame to impose the flatness constraint
    if mean_subtraction:
        along_look_slope = along_look_slope - np.nanmean(along_look_slope)
        cross_look_slope = cross_look_slope - np.nanmean(cross_look_slope)
    return along_look_slope,  cross_look_slope
