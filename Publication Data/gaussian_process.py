# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:28:22 2026

@author: ChemeGrad2020
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from sklearn import gaussian_process
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score, root_mean_squared_error
import pickle
import scipy.stats as stats

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
nmr=pd.read_excel('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Supplementary Data - Copolymerization.xlsx', sheet_name='NMR', header =[0,1])
ir = pd.read_excel('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Supplementary Data - Copolymerization.xlsx', sheet_name='FTIR', header =[0])

print(ir.columns)
figureparam_all= pd.read_excel('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Supplementary Data - Copolymerization.xlsx', sheet_name='Parameters', header =[0,1])
print(figureparam_all.columns)
print(nmr['On-line NMR'])
#we want to combine all the conversion data that we have (nmr and IR) and use all of it to train our GPR model
df1 = nmr['On-line NMR']
df2 = nmr['External NMR']
all_data = pd.concat([df1, df2, ir], ignore_index=True)
all_data = all_data.dropna()
print(all_data)


#get the instantaneous values for the timepoints of all the data

figureparam= figureparam_all[['Run Time', 'Effective']]
figureparam.columns=figureparam.columns.droplevel(0)
print(figureparam)
light = np.interp(all_data['Time (adj) (s)']/60,figureparam['time (min)'], figureparam['light intensity (range 0-1)'])
mon_ratio = np.interp(all_data['Time (adj) (s)']/60,figureparam['time (min)'], figureparam['MA/DMA molar ratio'])
temp = np.interp(all_data['Time (adj) (s)']/60,figureparam['time (min)'], figureparam['T (K)'])
res_time = np.interp(all_data['Time (adj) (s)']/60,figureparam['time (min)'], figureparam['res.time (min)'])

#print(light)

## add mask for when monomer fractions are between 0.05 and 0.95
mask =(mon_ratio > 0.05) & (mon_ratio < 0.95)
# print(mask.sum())
# print((~mask).sum())

#build gaussian process regressor
X_vals = np.column_stack((res_time, mon_ratio, temp, light))
X_vals= X_vals[mask]
Y_vals = all_data[['MA Conv', 'DMA Conv']]
Y_vals=Y_vals.loc[mask].copy()
print('conversiondata', Y_vals)



# kernel = (
#     ConstantKernel(1.0, (1e-3, 1e3))
#     * Matern(
#         length_scale=[10, 5, 100, 0.2],
#         length_scale_bounds=(1e-3, 1e3),
#         nu=2.5
#     )
#     + WhiteKernel(noise_level=1e-3)
# )
# kernel = (
#     ConstantKernel(1.0, (1e-3, 1e3))
#     * Matern(
#         length_scale=[10, 5, 100, 20],
#         #length_scale_bounds=(1e-3, 1e3),
#         nu=1.5
#     )
#     + WhiteKernel(noise_level_bounds=(1e-10,1e7))
# )
# gpr_base = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_targets=2, n_restarts_optimizer=5, random_state=30, normalize_y=True)
# gpr=gpr_base
# # gpr=TransformedTargetRegressor(regressor=gpr_base, transformer=LogitTransformer(low=0.0, high=1.0))
# gpr.fit(X_vals,Y_vals)

# with open('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Models/copolymer-NMR-IR-Conversion_masked-0.05.pkl','wb') as file:
#     pickle.dump(gpr, file)

with open('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/Models/copolymer-NMR-IR-Conversion_masked-0.05.pkl','rb') as file:
    gpr=pickle.load(file)
    
#build what data we want to evaluate
Temp_val = np.full(60,310)
light_val = np.full(60,0.4)
# res_time_val = np.full(60, 6)
# ma_dma_ratio = np.linspace(0.1, 0.9, num=60)
res_time_val = np.linspace(2,8, num=60)
ma_dma_ratio = np.full(60, 0.75)
print(np.column_stack((res_time_val, ma_dma_ratio,Temp_val,light_val)))

outputs, std=gpr.predict(np.column_stack((res_time_val, ma_dma_ratio,Temp_val,light_val)), return_std=True)

# print(outputs)
# print(std)
fig, ax=plt.subplots(1,1,layout='constrained',sharex=True)
ax1=ax
ax1.errorbar(res_time_val, outputs[:,0], yerr=std[:,0], color='black', capsize=3)
ax1.scatter(res_time_val, outputs[:,0],color='#a47b31',marker='.',s=100,label='MA Conversion',zorder=5)
ax1.errorbar(res_time_val, outputs[:,1], yerr=std[:,1], color='#085a31', capsize=3)
ax1.scatter(res_time_val, outputs[:,1],color = '#085a31', marker ='o', label = 'DMA Conversion')

ax1.legend(fontsize=16)
ax1.set_xlabel('Residence Time (min)',fontsize=16)
ax1.set_ylabel('Conversion', fontsize=16)
ax1.set_ylim(0,1)
plt.savefig('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/gpr-conversion-kernel_nmr_data_witherror_T=310_light=0.4_restime=2-8_masked_0.05.pdf',dpi=300,bbox_inches='tight')

#parity plot for conversion data
outputs_parity, std_parity = gpr.predict(X_vals, return_std=True)
print(outputs_parity)
r2_MA= r2_score(Y_vals['MA Conv'], outputs_parity[:,0])
r2_DMA = r2_score(Y_vals['DMA Conv'], outputs_parity[:,1])
rmse_MA = root_mean_squared_error(Y_vals['MA Conv'], outputs_parity[:,0])
rmse_DMA = root_mean_squared_error(Y_vals['DMA Conv'], outputs_parity[:,1])
print(rmse_MA, rmse_DMA)
fig, ax=plt.subplots(1,1,layout='constrained',sharex=True)
ax1=ax
ax1.errorbar(Y_vals['MA Conv'], outputs_parity[:,0], yerr=std_parity[:,0], color='black', capsize=3, linestyle='None')
ax1.scatter(Y_vals['MA Conv'], outputs_parity[:,0], facecolor=yellow[2],edgecolor=yellow[3],marker='.',s=100,zorder=5, label= f'MA Conversion\n$R^2$ Score: {r2_MA:0.3f}')
ax1.errorbar(Y_vals['DMA Conv'], outputs_parity[:,1], yerr=std_parity[:,1], color='#085a31', capsize=3, linestyle='None')
ax1.scatter(Y_vals['DMA Conv'], outputs_parity[:,1], color='#085a31',marker='o',zorder=5, label= f'DMA Conversion\n$R^2$ Score: {r2_DMA:0.3f}')
ax1.legend(fontsize=11)
plt.axline((0,0), slope=1, color = 'black', linestyle = '--')
ax1.set_xlabel(r'Conversion (Experimental)',fontsize=11)
ax1.set_ylabel(r'Conversion (Model)', fontsize=11)
plt.savefig('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/gpr-copolymer-parity_plot_masked_0.05.pdf',dpi=300,bbox_inches='tight')

#look at response of each conversion to feed comp and temperature
cbar_yellowgreen=LinearSegmentedColormap.from_list('yellowgreen',['#624a31','#a47b31','#d5c552','#ffe66a']+green,N=250)
cbar_pink=LinearSegmentedColormap.from_list('pink',pinks+['#40111f'],N=250)
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
ma_dma_ratio = np.full(len(X.ravel()), 0.5)
# Temp_val = np.full(len(X.ravel()),315)

outputs_demo, std_demo = gpr.predict(np.column_stack((Y.ravel(), ma_dma_ratio,Z.ravel(),X.ravel())),return_std=True)

Z_MA=outputs_demo[:,0]
Z_DMA=outputs_demo[:,1]

fig = plt.figure()
ax1=fig.add_subplot(projection='3d')
p=ax1.scatter3D(X.ravel(),Z.ravel(),Y.ravel(),c=Z_DMA,cmap=cbar_pink,alpha=0.6)
cbar1=fig.colorbar(p,ax=ax1, shrink=0.5,location='top',orientation='horizontal')
cbar1.set_label(label='DMA Conversion',fontsize=10)
# ax1.set_xlabel('LED Instensity\n(Fraction of Maximum)', size=10)
# ax1.set_ylabel('Temperature\n(K)', size=10)
# ax1.set_zlabel('Residence Time\n(min)', size=10)
ax1.set_box_aspect(None, zoom=0.8)
# ax1.plot_surface(X,Y,Z_DMA,color='#085a31', alpha=0.4)
print(fig.get_size_inches())
plt.savefig('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/gpr-copolymer-DMA_Conversion_masked_0.05.pdf',dpi=300,bbox_inches='tight')
plt.show()

fig = plt.figure()
ax1=fig.add_subplot(projection='3d')
cbar_pink=LinearSegmentedColormap.from_list('pink',pinks+['#40111f'],N=250)
p2=ax1.scatter3D(X.ravel(),Z.ravel(),Y.ravel(),c=Z_MA,cmap=cbar_yellowgreen,alpha=0.6)
cbar1=fig.colorbar(p2,ax=ax1, shrink=0.5,location='top',orientation='horizontal')
cbar1.set_label(label='MA Conversion',fontsize=10)
# ax1.set_xlabel('LED Intensity\n(Fraction of Maximum)', size=10)
# ax1.set_ylabel('Temperature\n(K)', size=10)
# ax1.set_zlabel('Residence Time\n(min)', size=10)
ax1.set_box_aspect(None, zoom=0.8)
# ax1.plot_surface(X,Y,Z_DMA,color='#085a31', alpha=0.4)
print(fig.get_size_inches())
plt.savefig('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/gpr-copolymer-MA_Conversion_masked_0.05.pdf',dpi=300,bbox_inches='tight')
plt.show()

print(gpr.kernel_.get_params())
#replicate Dylan's conditions for reactivity ratio - fineman and ross method
feed_MA= np.linspace(0.1,0.9,50)
res_time_val = np.full(50,5)
light_val = np.full(50,0.8)
# ma_dma_ratio = np.full(len(X.ravel()), 0.5)
Temp_val = np.full(50,300)
output_GPR=gpr.predict(np.column_stack((res_time_val,feed_MA,Temp_val,light_val)))

conv_MA=output_GPR[:,0]
conv_DMA=output_GPR[:,1]
feed_DMA=1-feed_MA

f1=feed_MA
F1=(conv_MA*feed_MA)/(conv_MA*feed_MA+conv_DMA*feed_DMA)
f=feed_MA/(1-feed_MA)
F=F1/(1-F1)
x=f**2/F
y=(F-1)*f/F
result=stats.linregress(x,y)
print(result)

#make two figures for reactivity ratio determination
#first figure, model data vs fit
fig, ax=plt.subplots(1,1,layout='constrained',figsize=(4.5,3))
ax.plot(x,result.slope*x+result.intercept,color=yellow[2],linestyle='--',label=f'Reactivity Ratio Fit:\n$r_{{MA}}$ = {result.slope:0.2f}, $r_{{DMA}}$ = {-1*result.intercept:0.2f}')
ax.scatter(x, y, color='#085a31',marker='o',zorder=5, label= 'Gaussian Process Data')
ax.set_xlabel(r'$\frac{f_{{MA}}^2(1-F_{{MA}})}{(1-f_{{MA}})^2F_{{MA}}}$',fontsize=11)
ax.set_ylabel(r'$\frac{f_{{MA}}(2F_{{MA}}-1)}{(1-f_{{MA}})F_{{MA}}}$', fontsize=11)
ax.legend(fontsize=11)
plt.savefig('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/gpr-copolymer-reactivity_ratio_fitting_masked_0.05.pdf',dpi=300,bbox_inches='tight')
fig.show()


#look at how reactivity ratio changes with light intensity and temperature

def Fineman_Ross(residence, temp, light, model):
    feed_MA= np.linspace(0.1,0.9,50)
    res_time_val = np.full(50,residence)
    light_val = np.full(50,light)
    # ma_dma_ratio = np.full(len(X.ravel()), 0.5)
    Temp_val = np.full(50,temp)
    output_GPR=model.predict(np.column_stack((res_time_val,feed_MA,Temp_val,light_val)))

    conv_MA=output_GPR[:,0]
    conv_DMA=output_GPR[:,1]
    feed_DMA=1-feed_MA

    f1=feed_MA
    F1=(conv_MA*feed_MA)/(conv_MA*feed_MA+conv_DMA*feed_DMA)
    f=f1/(1-f1)
    F=F1/(1-F1)
    x=f**2/F
    y=(F-1)*f/F
    result=stats.linregress(x,y)
    return result

lights=[0.2,0.5,0.8]
temps=[290,300,310]

r_MA=np.zeros((3,3))
r_DMA=np.zeros((3,3))

for i in range(3):
    for j in range(3):
        data=Fineman_Ross(5, temps[j], lights[i],gpr)
        print(data.rvalue)
        r_MA[i,j]=data.slope
        r_DMA[i,j]=-1*data.intercept

# print(r_MA)
# print(r_DMA)

fig, ax = plt.subplots(layout="constrained")

# Color scales
norm_MA = Normalize(vmin=r_MA.min(), vmax=r_MA.max())
norm_DMA = Normalize(vmin=r_DMA.min(), vmax=r_DMA.max())

#color maps are cbar_pink (DMA) and cbar_yellowgreen (MA)
for i in range(3):
    for j in range(3):

        # Square coordinates
        x0 = j
        x1 = j + 1
        y0 = i
        y1 = i + 1

        # MA = lower-left triangle
        triangle_MA = Polygon(
            [[x0, y0], [x1, y0], [x0, y1]],
            facecolor=cbar_yellowgreen(norm_MA(r_MA[i, j])),
            edgecolor="black"
        )

        # DMAA = upper-right triangle
        triangle_DMA = Polygon(
            [[x1, y0], [x1, y1], [x0, y1]],
            facecolor=cbar_pink(norm_DMA(r_DMA[i, j])),
            edgecolor="black"
        )

        ax.add_patch(triangle_MA)
        ax.add_patch(triangle_DMA)
        # MA numerical value
        ax.text(
            x0 + 0.33,
            y0 + 0.33,
            f"{r_MA[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            path_effects=[
                pe.withStroke(linewidth=2, foreground="white")
            ]
        )

        # DMAA numerical value
        ax.text(
            x0 +0.67,
            y0 +0.67,
            f"{r_DMA[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            path_effects=[
                pe.withStroke(linewidth=2, foreground="white")
            ]
        )

# Put tick labels at cell centers
ax.set_xticks(np.arange(3) + 0.5)
ax.set_xticklabels(temps)

ax.set_yticks(np.arange(3) + 0.5)
ax.set_yticklabels(lights)


ax.set_xlabel("Temperature (K)", fontsize=11)
ax.set_ylabel("LED Light Intensity\n(Fraction of Maximum)",fontsize=11)
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

ax.set_aspect("equal")
sm_MA = ScalarMappable(norm=norm_MA, cmap=cbar_yellowgreen)
sm_DMA = ScalarMappable(norm=norm_DMA, cmap=cbar_pink)

cbar_MA = fig.colorbar(
    sm_MA,
    ax=ax,
    fraction=0.046,
    pad=0.04
)

cbar_MA.set_label(r"$r_{{MA}}$")

cbar_DMA = fig.colorbar(
    sm_DMA,
    ax=ax,
    fraction=0.046,
    pad=0.12
)

cbar_DMA.set_label(r"$r_{{DMA}}$")
print(fig.get_size_inches())
plt.savefig('C:/Users/ChemeGrad2020/Documents/Grad School/Research/Automated Synthesis Robot/gpr-copolymer-reactivity_ratio_responses_masked_0.05.pdf',dpi=300,bbox_inches='tight')

