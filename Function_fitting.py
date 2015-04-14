# coding: utf-8

# Import packages ##############################################################
################################################################################

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import math

# Import files #################################################################
################################################################################

#  Import the required files (the complete data set FITS and voronoi bin data).

os.chdir("/home/ppxrh/Zoo_catalogues/Week_9/FITS")

gal_data=fits.getdata("d20.fits",1)

os.chdir("/home/ppxrh/Zoo_catalogues/voronoi")

bins=np.load("vor_arm_z.npy") # Voronoi bin data from the voronoi fitting. 

cols=["t11_arms_number_a31_1_weighted_fraction",
      "t11_arms_number_a32_2_weighted_fraction",
      "t11_arms_number_a33_3_weighted_fraction",
      "t11_arms_number_a34_4_weighted_fraction",
      "t11_arms_number_a36_more_than_4_weighted_fraction",
      "t11_arms_number_a37_cant_tell_weighted_fraction",
      "PETROMAG_MR","R50_KPC","REDSHIFT_1"]

gal_tb=np.array([gal_data.field(c) for c in cols])

x_guides=np.log10([0.2,0.5,0.8])
y_guides=np.array([0,1])

# Set the bin limits here:

bins=bins.T

v_min=int(np.min(bins[:,0])) # Min and max values of bins. 
v_max=int(np.max(bins[:,0]))

# Flag array for votes not reaching the minimum vote fraction:

flag=np.zeros((6,len(gal_data.T)))

min_vf=bins[:,13]

for a in range(0,6):

    flag[a]=gal_tb[a] >= min_vf

# Add an indexing column to keep all galaxies in the correct order:

i=np.array([np.arange(0,len(bins))])

data=np.concatenate([(bins[:,0:7].T),gal_tb,flag,i])

# Function for plotting the raw data and assigning CF values. ##################
################################################################################

def plot_raw(D,plot,style):

    D_ord=np.argsort(D[a+7])
    
    D_r=np.array([D[a+7],D[a+16],D[22]])
    
    D_r=(D_r.T[D_ord]).T
    
    D_p=D_r
    
    D_p=np.concatenate([D_r,np.array([np.linspace(0,1,len(D_r.T))])])
    
    D_p=(D_p.T[D_p[1] == 1]).T
    
    D_p[0]=np.log10(D_p[0])
    
    if plot ==1:
    
        plt.plot(D_p[0],D_p[3],"-",color=style)
        
        plt.xlabel("$\log(v_f)$")
        plt.ylabel("Cumulative fraction")
        
        plt.ylim([0,1])
        
    return np.array([D_p[0],D_p[3],D_p[2]]) # Returned array has a log(vf),a CF
# and an index column.

# Function to fit a function to the data bin output from the raw plot function #
################################################################################

def plot_function(D,plot,style):

# If plot=1, the function is plotted.
    
    def f(x,k,c,L):
        
        L=1+math.exp(c)
        
        if L >=100:
            
            L=0 # L is limited to stop a value growing too large, particularly 
# for the case of 2 armed galaxies.
        
        return L/(1+np.exp(-k*x+c))
    
    popt,pcov=curve_fit(f,D[0],D[1],maxfev=1000000,p0=[0,0,0])
    
    popt[2]=1+math.exp(popt[1])
    
    x=np.linspace(-4,0,1000)
    
    if plot == 1:
        
        plt.plot(x,f(x,popt[0],popt[1],popt[2]),"--",color=style)
        
        plt.ylim([0,1])
    
    return(popt) # Returns the optimal fit parameters. 

# Output a fitted function for each of the bins, arm numbers and redshift bins.
################################################################################

plt.close("all")

plot=0

clr=[0,0,0]
clf=[1,0,0]

# Set up the array to write the parameters in to:

param_data=np.zeros((10000,8))

r=0

for v in range(v_min,v_max+1):
    
    data_plot=(data.T[(data[0] == v)]).T
    
    for a in range(0,6):
        
        z_min=int(np.min(data_plot[a+1]))
        z_max=int(np.max(data_plot[a+1]))
        
        clr=[0,0,1]
        
        clr_diff=(1/(z_max-z_min))
        
        for z in range(z_min,z_max+1):
            
            data_z=(data_plot.T[(data_plot[a+1] == z)]).T
            
            clr_z=[np.min(np.array([clr[0]+(z-1)*clr_diff,1])),
                   0,np.max(np.array([clr[2]-(z-1)*clr_diff,0]))]
    
            plt.subplot(2,3,a+1)

            n=plot_raw(data_z,plot,clr_z)
    
            p=plot_function(n,plot,clr_z)
        
            locals()["n_{}_{}".format(v,a)]=n
            locals()["p_{}_{}".format(v,a)]=p
            
            param_data[r,0:3]=[v,a,z]
            param_data[r,3:6]=np.mean(data_z[13:16],axis=1)
            param_data[r,6:]=[p[0],p[1]]
            
            r=r+1
    
if plot == 1:
    
    for a in range(0,6):
        
        plt.subplot(2,3,a+1)
        
        for g in range(0,3):
        
            plt.plot([x_guides[g],x_guides[g]],y_guides,color=[0,0,0],alpha=0.3)

    plt.show()
    
param_data=param_data[0:r,:]

# Output parameters for each bin:

# 0: v bin
# 1: a (arm number-1)
# 2: z bin
# 3: M_r (mean of bin)
# 4: R_50 (mean of bin)
# 5: redshift
# 6: k (fitted)
# 7: c (fitted)

# Save the fitted parameters to a numpy table. #################################
################################################################################

np.save("fixed_bin_size_params_2.npy",param_data)

################################################################################
################################################################################



