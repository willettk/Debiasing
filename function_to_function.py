
# coding: utf-8

# Import packages. #############################################################
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import math

os.chdir("/home/ppxrh/Zoo_catalogues/voronoi")

data=np.load("fixed_bin_size_params_2.npy")

v_min=int(np.min(data[:,0]))
v_max=int(np.max(data[:,0]))

# Define the fitting function. #################################################
################################################################################

def f_f(x,k,c): 
        
    L=1+math.exp(c)
        
    return L/(1+np.exp(-k*x+c))

# Fit a linear function of M,R and z to the bins. ##############################
################################################################################

params=np.zeros((6,9))

kmin=np.zeros((6,2))
kmax=np.zeros((6,2))
cmin=np.zeros((6,2))
cmax=np.zeros((6,2))

def f(x,A0,AM,AR,Az):
    return A0 + AM*x[0] + AR*x[1] + Az*x[2] # Function is linear in the 3 parameters with an offset. 

for a in range(0,6):
    
    data_arm=data[data[:,1] == a]
    
    M=data_arm[:,3]
    R=data_arm[:,4]
    redshift=data_arm[:,5]
    
    x=np.array([M,R,redshift])
    
    k=data_arm[:,6]
    c=data_arm[:,7]
    
    cmax[a,:]=data_arm[:,6:8][np.argmax(c)]
    cmin[a,:]=data_arm[:,6:8][np.argmin(c)]
    kmax[a,:]=data_arm[:,6:8][np.argmax(k)]
    kmin[a,:]=data_arm[:,6:8][np.argmin(k)]
    
    kp,kc=curve_fit(f,x,k,maxfev=1000) # Fit k and c to the parameters. 
    cp,cc=curve_fit(f,x,c,maxfev=1000)
    
    params[a,0]=a
    params[a,1:5]=kp
    params[a,5:]=cp

# Save the output parameters. ##################################################
################################################################################

np.save("kc_fit_params.npy",params)
np.save("cmin.npy",cmin)
np.save("cmax.npy",cmax)
np.save("kmin.npy",kmin)
np.save("kmax.npy",kmax)

################################################################################
################################################################################
