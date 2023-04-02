#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 20:51:54 2023

@author: nacer
"""

import numpy as np

def r_squared(x, y):
    residuals = np.array(y) - np.array(x)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)  
    r_sqr = 1 - (ss_res / ss_tot)
    
    return r_sqr
