#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import astropy.constants

light_speed=astropy.constants.c.value

def normalized_dipole_beam(theta, freq, length):
    """
    返回一个len(freq)行，len(theta)列的二维数组，其第(i,j)号元素的值等于频率freq[i]，方向theta[j]的normalized gain
    """
    theta=np.atleast_1d(theta)
    freq=np.atleast_1d(freq)
    lbd=light_speed/freq
    k=2*np.pi/lbd
    kl=k*length
    theta1=theta

    klct=np.cos(np.einsum('i,j->ij',kl/2,np.cos(theta1)))
    ckl=np.cos(np.einsum('ij,i->ij', np.ones_like(klct), kl/2))
    st=np.sin(np.einsum('ij,j->ij', np.ones_like(klct), theta1))
    b=(klct-ckl)/st
    b2=b**2
    b2st=b2*st
    norm=np.sum(b2st, axis=1)
    for i in range(0, b2.shape[0]):
        b2[i,:]/=norm[i]

    #b=((np.cos(kl/2*np.cos(theta1))-np.cos(kl/2))/np.sin(theta1))**2
    #return b/np.sum(b*np.sin(theta1))
    return b2
