# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:43:03 2026

@author: ChemeGrad2020
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import gaussian_process
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score
import pickle
# from sklearn.preprocessing import StandardScalar

mpl.rcParams["font.family"]='Arial'

pinks=['#ee9cac', '#cd6a83','#b4414a','#7b2929']
green=['#9cde83','#399c52','#085a31']
yellow=['#fff6ac','#ffe66a','#d5c552','#a47b31','#624a31']

class LogitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high

    def fit(self, Y, y=None):
        return self

    def transform(self, Y):
        # Scale to strict (0, 1) to avoid log(0) or log(inf)
        eps = 1e-7
        Y_scaled = (Y - self.low) / (self.high - self.low)
        Y_scaled = np.clip(Y_scaled, eps, 1 - eps)
        return np.log(Y_scaled / (1 - Y_scaled))

    def inverse_transform(self, Y):
        # Sigmoid function maps back to (low, high)
        sigmoid = 1 / (1 + np.exp(-Y))
        return sigmoid * (self.high - self.low) + self.low

# putting together gaussian process regressor for models

#start with copolymerization for discussion, just the nmr data
nmr=pd.read_excel('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Supplementary Data - Homopolymerization.xlsx', sheet_name='NMR', header =[0,1])
ir = pd.read_excel('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Supplementary Data - Homopolymerization.xlsx', sheet_name='FTIR', header =[0])
gpc = pd.read_excel('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Supplementary Data - Homopolymerization.xlsx', sheet_name='GPC (THF)', header =[0,1])
print(ir.columns)
figureparam_all= pd.read_excel('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Supplementary Data - Homopolymerization.xlsx', sheet_name='Parameters', header =[0,1])
print(figureparam_all.columns)
print(nmr['On-line NMR'])
#we want to combine all the conversion data that we have (nmr and IR) and use all of it to train our GPR model
df1 = nmr['On-line NMR']
df2 = nmr['External NMR']
gpc = gpc['THF GPC, PS Equivalent Values']
all_data = pd.concat([df1, df2, ir], ignore_index=True)
all_data = all_data.dropna()
print(all_data)


#get the instantaneous values for the timepoints of all the data, and scale them for their periodicity to help with implementing a periodic kernel

figureparam= figureparam_all[['Run Time', 'Effective']]
figureparam.columns=figureparam.columns.droplevel(0)
print(figureparam)

light = np.interp(all_data['Time (adj) (s)']/60,figureparam['time (min)'], figureparam['light intensity (range 0-1)'])
ca_ratio = np.interp(all_data['Time (adj) (s)']/60,figureparam['time (min)'], figureparam['cat/MA molar ratio'])
temp = np.interp(all_data['Time (adj) (s)']/60,figureparam['time (min)'], figureparam['T (K)'])
res_time = np.interp(all_data['Time (adj) (s)']/60,figureparam['time (min)'], figureparam['res.time (min)'])

#print(light)

kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * Matern(
        length_scale=[10, 5, 100, 20],
        #length_scale_bounds=(1e-3, 1e3),
        nu=1.5
    )
    + WhiteKernel(noise_level_bounds=(1e-10,1e7))
)
#build gaussian process regressor

X_vals = np.column_stack((res_time, ca_ratio, temp, light))
Y_vals = all_data[['MA Conv']]
print('conversiondata', X_vals)
# gpr_base = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=30, normalize_y=True)
# gpr=gpr_base
# # gpr=TransformedTargetRegressor(regressor=gpr_base, transformer=LogitTransformer(low=0.0, high=1.0))
# gpr.fit(X_vals,Y_vals)
# with open('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Models/homopolymer-NMR-IR-Conversion.pkl','wb') as file:
#     pickle.dump(gpr, file)

with open('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Models/homopolymer-NMR-IR-Conversion.pkl','rb') as file:
    gpr=pickle.load(file)
#build what data we want to evaluate
Temp_val = np.full(60,310)
light_val = np.full(60,0.4)
ca_ratio_val = np.full(60, 5*10e-5)
res_time_val = np.linspace(2,8, num=60)

# print(np.column_stack((res_time_val, ca_ratio_val,Temp_val,light_val)))

outputs, std=gpr.predict(np.column_stack((res_time_val, ca_ratio_val,Temp_val,light_val)), return_std=True)

# print(outputs)
# print(std)
fig, ax=plt.subplots(1,1,layout='constrained',sharex=True)
ax1=ax
ax1.errorbar(res_time_val, outputs[:], yerr=std[:], color='black', capsize=5)
ax1.scatter(res_time_val, outputs[:], color='#a47b31',marker='.',s=100, label='MA Conversion',zorder=5)
# ax1.errorbar(res_time_val, outputs[:,1], yerr=std[:,1], color='#085a31', capsize=5)
# ax1.scatter(res_time_val, outputs[:,1],color = '#085a31', marker ='o', label = 'DMA Conversion')

ax1.legend(fontsize=11)
ax1.set_xlabel('Residence Time',fontsize=11)
ax1.set_ylabel('Conversion', fontsize=11)
ax1.set_ylim(0,1)
# plt.savefig('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/gpr-homopolymer_conversion-perkernel_all_data_witherror_T=310_light=0.4_restime=2-8.pdf',dpi=300,bbox_inches='tight')
fig.show()

#now do all of this again for the GPC Data
light_gpc = np.interp(gpc['Time (adj) (s)']/60,figureparam['time (min)'], figureparam['light intensity (range 0-1)'])
ca_ratio_gpc = np.interp(gpc['Time (adj) (s)']/60,figureparam['time (min)'], figureparam['cat/MA molar ratio'])
temp_gpc = np.interp(gpc['Time (adj) (s)']/60,figureparam['time (min)'], figureparam['T (K)'])
res_time_gpc = np.interp(gpc['Time (adj) (s)']/60,figureparam['time (min)'], figureparam['res.time (min)'])

#print(light)

kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * Matern(
        length_scale=[10, 5, 100, 20],
        #length_scale_bounds=(1e-3, 1e3),
        nu=1.5
    )
    + WhiteKernel(noise_level_bounds=(1e-10,1e7))
)
#build gaussian process regressor

X_vals_gpc = np.column_stack((res_time_gpc, ca_ratio_gpc, temp_gpc, light_gpc))
Y_vals_gpc = gpc[['Mn']]
print('conversiondata', Y_vals_gpc)
# gpr_base_gpc = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=10, normalize_y=True)
# gpr_gpc=gpr_base_gpc
# # gpr=TransformedTargetRegressor(regressor=gpr_base, transformer=LogitTransformer(low=0.0, high=1.0))
# gpr_gpc.fit(X_vals_gpc,Y_vals_gpc)
# with open('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Models/homopolymer-NMR-IR-GPC.pkl','wb') as file:
#     pickle.dump(gpr_gpc, file)

with open('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Models/homopolymer-NMR-IR-GPC.pkl','rb') as file:
    gpr_gpc=pickle.load(file)


# print(np.column_stack((res_time_val, ca_ratio_val,Temp_val,light_val)))

outputs_gpc, std_gpc=gpr_gpc.predict(np.column_stack((res_time_val, ca_ratio_val,Temp_val,light_val)), return_std=True)

# print(outputs_gpc)
# print(std_gpc)
fig, ax=plt.subplots(1,1,layout='constrained',sharex=True)
ax1=ax
ax1.errorbar(res_time_val, outputs_gpc[:], yerr=std_gpc[:], color='black', capsize=3)
ax1.scatter(res_time_val, outputs_gpc[:], facecolor=yellow[2],edgecolor=yellow[3],marker='.',s=100, label=r'$M_n$ from GPC',zorder=5)
ax1.errorbar(res_time_val, outputs[:]*200*86.09, yerr=std[:]*200*86.09, color ='#085a31', capsize=3 )
ax1.scatter(res_time_val, outputs[:]*200*86.09, color='#085a31',marker='o', label=r'Theoretical $M_n$',zorder=4)
# ax1.errorbar(res_time_val, outputs[:,1], yerr=std[:,1], color='#085a31', capsize=5)
# ax1.scatter(res_time_val, outputs[:,1],color = '#085a31', marker ='o', label = 'DMA Conversion')

ax1.legend(fontsize=11)
ax1.set_xlabel('Flow Time',fontsize=11)
ax1.set_ylabel(r'$M_n$ (Da)', fontsize=11)
# ax1.set_ylim(0,1)
# plt.savefig('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/gpr-homopolymer_conversion-GPCvsTheoretical_witherror_T=310_light=0.4_restime=2-8.pdf',dpi=300,bbox_inches='tight')
fig.show()

#compare model to actual data points - parity plot
outputs_conv, std_conv = gpr.predict(X_vals, return_std=True)
print(Y_vals)
r2= r2_score(Y_vals, outputs_conv[:])
fig, ax=plt.subplots(1,1,layout='constrained',sharex=True)
ax1=ax
ax1.errorbar(Y_vals, outputs_conv[:], yerr=std_conv[:], color='black', capsize=3, linestyle='None')
ax1.scatter(Y_vals, outputs_conv[:], facecolor=yellow[2],edgecolor=yellow[3],marker='.',s=100,zorder=5, label= f'$R^2$ Score: {r2:0.3f}')
ax1.legend(fontsize=11)
plt.axline((0,0), slope=1, color = 'black', linestyle = '--')
ax1.set_xlabel('Conversion (Experimental)',fontsize=11)
ax1.set_ylabel('Conversion (Model)', fontsize=11)
plt.savefig('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/gpr-homopolymer_conversion-parity_plot.pdf',dpi=300,bbox_inches='tight')

outputs_gpc_parity, std_gpc_parity = gpr_gpc.predict(X_vals_gpc, return_std=True)
r2= r2_score(Y_vals_gpc, outputs_gpc_parity[:])
fig, ax=plt.subplots(1,1,layout='constrained',sharex=True)
ax1=ax
ax1.errorbar(Y_vals_gpc, outputs_gpc_parity[:], yerr=std_gpc_parity[:], color='black', capsize=3, linestyle='None')
ax1.scatter(Y_vals_gpc, outputs_gpc_parity[:], color='#085a31',marker='o',zorder=5, label= f'$R^2$ Score: {r2:0.3f}')
ax1.legend(fontsize=11)
plt.axline((0,0), slope=1, color = 'black', linestyle = '--')
ax1.set_xlabel(r'Conversion (Experimental)',fontsize=11)
ax1.set_ylabel(r'Conversion (Model)', fontsize=11)
plt.savefig('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/gpr-homopolymer_gpc-parity_plot.pdf',dpi=300,bbox_inches='tight')


#evaluation of conversion from model results

cbar_yellowgreen=LinearSegmentedColormap.from_list('yellowgreen',['#624a31','#a47b31','#d5c552','#ffe66a']+green,N=250)
Temp_val = np.linspace(290,320,num=5)
light_val = np.linspace(0.2,0.8,num=10)
# res_time_val = np.full(60, 6)
# ma_dma_ratio = np.linspace(0.2, 0.8, num=60)
res_time_val = np.linspace(2,10,num=7)
# ma_dma_ratio = np.full(60, 0.75)
# X=Temp_val
Y=res_time_val
X=light_val
Z=Temp_val
X,Y,Z = np.meshgrid(light_val, res_time_val,Temp_val)
# res_time_val = np.full(len(X.ravel()),8)
# light_val = np.full(len(X.ravel()),0.4)
ca_ratio_val = np.full(len(X.ravel()), 5*10e-5)
# Temp_val = np.full(len(X.ravel()),315)

outputs_demo, std_demo = gpr.predict(np.column_stack((Y.ravel(), ca_ratio_val,Z.ravel(),X.ravel())),return_std=True)

Z_MA=outputs_demo[:]
# Z_DMA=outputs_demo[:,1]

fig = plt.figure()
ax1=fig.add_subplot(projection='3d')
p=ax1.scatter3D(X.ravel(),Z.ravel(),Y.ravel(),c=Z_MA,cmap=cbar_yellowgreen,alpha=0.6)
cbar1=fig.colorbar(p,ax=ax1, shrink=0.5,location='top',orientation='horizontal')
cbar1.set_label(label='MA Conversion',fontsize=10)
# ax1.set_xlabel('LED Instensity\n(Fraction of Maximum)', size=10)
# ax1.set_ylabel('Temperature\n(K)', size=10)
# ax1.set_zlabel('Residence Time\n(min)', size=10)
ax1.set_box_aspect(None, zoom=0.8)
# ax1.plot_surface(X,Y,Z_DMA,color='#085a31', alpha=0.4)
print(fig.get_size_inches())
plt.savefig('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/gpr-homopolymer-MA_Conversion.pdf',dpi=300,bbox_inches='tight')
plt.show()



