# coding: utf-8

# 3rd to run

# Import packages ##############################################################
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from astropy.io import fits
import math

# Load the required data. ######################################################
################################################################################

gal_data=fits.getdata("fits/d20.fits",1) # Galaxy zoo data.

params=np.load("npy/kc_fit_params.npy") # Parameters from the function fitting.

cmin=np.load("npy/cmin.npy")
cmax=np.load("npy/cmax.npy")
kmin=np.load("npy/kmin.npy")
kmax=np.load("npy/kmax.npy")

# Have included some limits to the data- for each arm number the minimum and maximum c and k values are set as the limits to the functions. 

cols=["t11_arms_number_a31_1_weighted_fraction",
      "t11_arms_number_a32_2_weighted_fraction",
      "t11_arms_number_a33_3_weighted_fraction",
      "t11_arms_number_a34_4_weighted_fraction",
      "t11_arms_number_a36_more_than_4_weighted_fraction",
      "t11_arms_number_a37_cant_tell_weighted_fraction",
      "PETROMAG_MR","R50_KPC","REDSHIFT_1"]

gal_tb=np.array([gal_data.field(c) for c in cols])

i=np.array([np.arange(0,len(gal_tb.T))])

gal_tb=np.concatenate([gal_tb,i]) # i is an index column.

# Define functions for getting log(vf) (x) from CF (y) and v.v. ################
################################################################################

def f(x,k,c): 
        
    L=1+np.exp(c)
        
    return L/(1+np.exp(-k*x+c))

def i_f(y,k,c):
    
    L=1+np.exp(c)
    
    return -(1/k)*(np.log((L/y)-1)-c)


# Debiasing procedure. #########################################################
################################################################################

debiased=np.zeros((6,len(gal_tb.T)))

z_base=0.02 # Low redshift that functions are corrected to. 

# Each galaxy gets a function fit to its M,R and z parameters, which are scaled 
# to the equivalent M and r functions at low z.

for a in range(0,6):
    
    p=params[a]
    
    k=params[a,1]+params[a,2]*gal_tb[6]+params[a,3]*gal_tb[7]+params[a,4]*gal_tb[8]
    c=params[a,5]+params[a,6]*gal_tb[6]+params[a,7]*gal_tb[7]+params[a,8]*gal_tb[8]
    
    kb=params[a,1]+params[a,2]*gal_tb[6]+params[a,3]*gal_tb[7]+params[a,4]*z_base
    cb=params[a,5]+params[a,6]*gal_tb[6]+params[a,7]*gal_tb[7]+params[a,8]*z_base
    
    kc=np.array([k,c,kb,cb])
    
    # Section for dealing with any functions outside the k and c limits. #
        
    kh=kc[0] > kmax[a,0]
    kl=kc[0] < kmin[a,0]
    ch=kc[1] > cmax[a,1]
    cl=kc[1] < cmin[a,1]
    
    kbh=kc[2] > kmax[a,0]
    kbl=kc[2] < kmin[a,0]
    cbh=kc[3] > cmax[a,1]
    cbl=kc[3] < cmin[a,1]
    
    kc=kc.T

    kc[:,0:2][kh]=kmax[a,:]
    kc[:,0:2][kl]=kmin[a,:]
    kc[:,0:2][ch]=cmax[a,:]
    kc[:,0:2][cl]=cmin[a,:]
    
    kc[:,2:][kbh]=kmax[a,:]
    kc[:,2:][kbl]=kmin[a,:]
    kc[:,2:][cbh]=cmax[a,:]
    kc[:,2:][cbl]=cmin[a,:]
    
    kc=kc.T
    
    ######################################################################
    
    a_d=np.array([gal_tb[-1],gal_tb[a],kc[0],kc[1],kc[2],kc[3]])
    
    a_d=(a_d.T[a_d[1] > 0]).T
    
    a_d[1]=np.log10(a_d[1])
    
    CF=f(a_d[1],a_d[2],a_d[3])
    
    lvd=i_f(CF,a_d[4],a_d[5])
    
    vdeb=10**(lvd)
    
    i=a_d[0].astype(int)
    
    debiased[a][i]=vdeb

# Summing/normalisation if wanted. #############################################
################################################################################

sums=np.sum(debiased,axis=0)

# Plot debiased values vs. raw values for comparison. Blue -> red with z. ######
################################################################################

plt.figure(1)

raw=gal_tb[0:6]

z=gal_tb[-2]

z=np.log10(z)
    
scaled_z = (z - z.min()) / z.ptp()
colors = plt.cm.coolwarm(scaled_z)

pld=debiased

for a in range(0,6):
    
    plt.figure(1)
    plt.subplot(2,3,a+1)
    
    plt.scatter(raw[a],pld[a],marker="+",s=30,c=colors)
    
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.plot([0,1],[0,1],"k-")
    
    plt.xlabel(r"$f_v$")
    plt.ylabel(r"$f_{debiased}$")
    
plt.show()

# Plot histogram of sums of f_v. ###############################################
################################################################################

plt.figure(2)

plt.hist(sums,bins=30)

plt.xlabel(r"$\Sigma f_v$")
plt.ylabel(r"$N_{gal}$")

plt.show()

###############################################################################
################################################################################



