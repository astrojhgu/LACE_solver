#!/usr/bin/env python

import numpy as np
import astropy.constants

light_speed=astropy.constants.c.value

def normalized_dipole_beam(theta, freq, length):
    lbd=light_speed/freq
    k=2*np.pi/lbd
    kl=k*length
    theta1=theta
    b=((np.cos(kl/2*np.cos(theta1))-np.cos(kl/2))/np.sin(theta1))**2
    return b/np.sum(b*np.sin(theta1))


