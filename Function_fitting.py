# coding: utf-8

# 1st to run

# Import packages ##############################################################
################################################################################

from __future__ import division
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

n_morph = 6
tasklabels = {0:'1',
            1:'2',
            2:'3',
            3:'4',
            4:'5+',
            5:'??'}

# Import files #################################################################
################################################################################

#  Import the required files (the complete data set FITS and Voronoi bin data).

def load_data():

    gal_data=fits.getdata("fits/d20.fits",1)
    bins=np.load("npy/vor_arm_z.npy") # Voronoi bin data from the voronoi fitting. 
    
    cols=["t11_arms_number_a31_1_weighted_fraction",
          "t11_arms_number_a32_2_weighted_fraction",
          "t11_arms_number_a33_3_weighted_fraction",
          "t11_arms_number_a34_4_weighted_fraction",
          "t11_arms_number_a36_more_than_4_weighted_fraction",
          "t11_arms_number_a37_cant_tell_weighted_fraction",
          "PETROMAG_MR","R50_KPC","REDSHIFT_1"]
    
    # Limit the working dataset to only the columns we need (morphology + binning parameters)
    gal_tb=np.array([gal_data.field(c) for c in cols])
    
    bins=bins.T         # Dimensions are now (30521,15); rows = n_galaxies, columns = ??
    
    # Flag array for votes not reaching the minimum vote fraction:
    
    flag=np.zeros((n_morph,len(gal_data.T))) # Empty array
    
    min_vf=bins[:,13] # The 13th (last but one) slice in the Voronoi bins
    
    for a in range(0,n_morph):
    
        flag[a] = gal_tb[a] >= min_vf
    
    # Add an indexing column to keep all galaxies in the correct order:
    
    i=np.array([np.arange(0,len(bins))])
    
    data=np.concatenate([(bins[:,0:7].T),gal_tb,flag,i])

    return data,bins

# Function for plotting the raw data and assigning cumulative fraction values. #
################################################################################

def plot_raw(D_p,a,style):

    plt.plot(D_p[0],D_p[3],"-",color=style,lw=2)
    
    return None # Returned array has a log(vf),a CF
                # and an index column.

# Function to fit a function to the data bin output from the raw plot function #
################################################################################

def f(x,k,c,L):
    
    L=1+math.exp(c)
    
    if L >=100:
        
        L=0 # L is limited to stop a value growing too large, particularly 
            # for the case of 2-armed galaxies.
    
    return L/(1+np.exp(-k*x+c))
    
def plot_function(x,popt,style):

    plt.plot(x,f(x,popt[0],popt[1],popt[2]),"--",color=style,lw=0.5)
    
    return None # Returns the optimal fit parameters. 

# Output a fitted function for each of the bins, arm numbers and redshift bins.
################################################################################

def fit_function(data,bins,plot=True,savefig=False):

    clr=[0,0,0]
    clf=[1,0,0]
    
    # Set up the array to write the parameters in to:
    
    param_data=np.zeros((10000,8))
    
    # Set the bin limits here:
    v_min=int(np.min(bins[:,0])) #  1
    v_max=int(np.max(bins[:,0])) # 28
    
    for r,v in enumerate(range(v_min,v_max+1)):
        
        data_plot=(data.T[(data[0] == v)]).T
        
        for a in range(0,n_morph):
            
            z_min=int(np.min(data_plot[a+1]))
            z_max=int(np.max(data_plot[a+1]))
            
            clr=[0,0,1]
            
            clr_diff=(1/(z_max-z_min))
            
            for z in range(z_min,z_max+1):
                
                data_z=(data_plot.T[(data_plot[a+1] == z)]).T
                
                clr_z=[np.min(np.array([clr[0]+(z-1)*clr_diff,1])),
                       0,np.max(np.array([clr[2]-(z-1)*clr_diff,0]))]

                # Old plot_raw output
                D_ord=np.argsort(data_z[a+7])
                D_r=np.array([data_z[a+7],data_z[a+16],data_z[22]])
                D_r=(D_r.T[D_ord]).T
                D_p=D_r
                D_p=np.concatenate([D_r,np.array([np.linspace(0,1,len(D_r.T))])])
                D_p=(D_p.T[D_p[1] == 1]).T
                D_p[0]=np.log10(D_p[0])

                n = np.array([D_p[0],D_p[3],D_p[2]])
    
                # Old plot_function output
                popt,pcov=curve_fit(f,n[0],n[1],maxfev=1000000,p0=[0,0,0])
                popt[2]=1+math.exp(popt[1])
                x=np.linspace(-4,0,1000)
                p = popt
    
                if plot:
                    plt.subplot(2,3,a+1)
                    plot_raw(D_p,a,clr_z)
                    plot_function(x,popt,clr_z)
            
                locals()["n_{}_{}".format(v,a)]=n
                locals()["p_{}_{}".format(v,a)]=p
                
                param_data[r,0:3]=[v,a,z]
                param_data[r,3:6]=np.mean(data_z[13:16],axis=1)
                param_data[r,6:]=[p[0],p[1]]

            if plot and r==0:
                plt.tick_params(axis='both',labelsize=10)
                plt.xticks(np.arange(5)-4)
                plt.text(-3.9,0.9,r'$N_{arms}=$%s' % tasklabels[a],fontsize=10,ha='left')
                plt.ylim([0,1])
    
                if a > 2:
                    plt.xlabel("$\log(v_f)$")

                if a == 0 or a == 3:
                    plt.ylabel("Cumulative fraction")
    

    param_data=param_data[0:r,:]

    if plot:
        if savefig:
            plt.savefig('plots/Function_fitting.pdf', dpi=200)
        else:
            plt.show()

    return None

def plot_fit_function():

    x_guides=np.log10([0.2,0.5,0.8])
    y_guides=np.array([0,1])
    
    for a in range(0,n_morph):
        
        plt.subplot(2,3,a+1)
        
        for xg in x_guides:
        
            plt.plot([xg,xg],y_guides,color=[0,0,0],alpha=0.3)
    
    return None
        
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

def save_fit_function(param_data):

    np.save("npy/fixed_bin_size_params_2.npy",param_data)

    return None

################################################################################
################################################################################

if __name__ == "__main__":

    data,bins = load_data()
    param_data = fit_function(data,bins,plot=False,savefig=True)
    save_fit_function(param_data)

