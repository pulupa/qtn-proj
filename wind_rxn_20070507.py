# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:17:14 2015

@author: pulupa
"""
from scipy.io.idl import readsav
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import bisect
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline

import os.path

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['xtick.direction'] = 'out'; rcParams['ytick.direction'] = 'out';

matplotlib.rc('text', usetex=True)

qtn_dir = '/Users/pulupa/Documents/qtn-proj/'

sys.path.append(qtn_dir)

from qtn.bimax import BiMax

#%%
"""
Load IDL save file data
"""

tnr_dat = readsav(qtn_dir + 'data/tnr_dat_20070507.sav')

tnr_lfnoise_ind = [10379, 10380, 10381, 10421, 10422]

tnr_dts = []
for tnr_unix in (tnr_dat['x']):
    tnr_dts.append(datetime.datetime.utcfromtimestamp(tnr_unix))
    
tnr_lfnoise_dts = [tnr_dts[i] for i in tnr_lfnoise_ind]    
    
tnr_f = tnr_dat['v']

nn_dat = readsav(qtn_dir + 'data/nn_dat_20070507.sav')

nn_dts = []
for nn_unix in (nn_dat['x']):
    nn_dts.append(datetime.datetime.utcfromtimestamp(nn_unix))
    
nn_ne = nn_dat['y']

#%% 
"""
Input datetimes
"""

strt_t = datetime.datetime(2007,5,7,12,45,00) # overall plot times
stop_t = datetime.datetime(2007,5,7,12,49,30)

evdf_times = [datetime.datetime(2007,5,7,12,45,26), # 3 EVDFs
              datetime.datetime(2007,5,7,12,47,03),
              datetime.datetime(2007,5,7,12,48,41)]

exhaust_bounds = [datetime.datetime(2007,5,7,12,46,30),
                  datetime.datetime(2007,5,7,12,48,15)]

exhaust_bounds = [[datetime.datetime(2007,5,7,12,45,03),
                   datetime.datetime(2007,5,7,12,46,35)],
                  [datetime.datetime(2007,5,7,12,46,55),
                   datetime.datetime(2007,5,7,12,48,10)],
                  [datetime.datetime(2007,5,7,12,48,30),
                   datetime.datetime(2007,5,7,12,49,27)],]


tnr_inds = [tnr_dts.index(min(tnr_dts, key=lambda dt:abs(dt-evdf_time))) \
    for evdf_time in evdf_times]

tnr_v2 = np.square(tnr_dat['y'][:,tnr_inds])

tnr_times = [tnr_dts[ind] for ind in tnr_inds]

nn_inds = [nn_dts.index(min(nn_dts, key=lambda dt:abs(dt-evdf_time))) \
    for evdf_time in evdf_times]

nn_ne_evdfs = nn_ne[nn_inds]

nn_time_evdfs = [nn_dts[ind] for ind in nn_inds]

#%%
"""
Wind antenna parameters
"""

#ant_len = 50.      # m (monopole) 
ant_len = 30.      # m (monopole) 
ant_rad = 1.9e-4  # m
base_cap = 20e-12 # Farad
fbins = np.array([4000*1.0445**i for i in range(96)])

#%%
"""
Solar wind parameters
"""

qtn_txt = 'data/wi_sw_evdf_20070507_qtn3.txt'

evdf_dat = np.genfromtxt(qtn_dir + qtn_txt, dtype = None, names = True)

vsw = evdf_dat['vsw'] * 1.e3
tc = evdf_dat['tc'] 
t = evdf_dat['t']
n = evdf_dat['n']
ne = evdf_dat['nelec']
tp = evdf_dat['tp']
fpe = [8980. * np.sqrt(ne)]

#%%
"""
Calculation (if needed) and Plots
"""

for i, tnr_time in enumerate(tnr_times):

    time_str = tnr_times[i].strftime("%Y%m%d_%H%M%S")

    qtn_file = qtn_dir + 'data/qtn_spec' + time_str
    qtn_plot = qtn_dir + 'plots_20070507/qtn_spec' + time_str

    p = BiMax(ant_len, ant_rad, base_cap)
    
    if not os.path.isfile(qtn_file + '.npz'):

        proton_noise = \
        np.array([p.proton(f,ne[i],n[i],t[i],tp[i],tc[i],vsw[i]) \
        for f in fbins])
        
        electron_noise = \
        np.array([p.electron_noise(f,ne[i],n[i],t[i],tp[i],tc[i],vsw[i]) \
        for f in fbins])
        
        gain_shot = \
        np.array([np.array(p.gain_shot(f,ne[i],n[i],t[i],tp[i],tc[i],vsw[i])) \
        for f in fbins])
        
        gain = gain_shot[:,0]
        shot_noise = gain_shot[:, 1]
        
        np.savez(qtn_file, p_noise=proton_noise, \
        e_noise = electron_noise, s_noise = shot_noise, gain = gain)    
        
    else:
        
        qtn_saved_data = np.load(qtn_file + '.npz')
        
        proton_noise = qtn_saved_data['p_noise']        
        electron_noise = qtn_saved_data['e_noise']        
        shot_noise = qtn_saved_data['s_noise']        
        gain = qtn_saved_data['gain']        
     
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=[4, 4])

    plt.plot(fbins/1000, \
        (electron_noise + proton_noise + shot_noise)/gain, label='QTN')
    plt.plot(fbins/1000, electron_noise/gain,'--', label='electron', dashes = (3,3))
    plt.plot(fbins/1000, proton_noise/gain, '-.', label='proton', dashes = (3,3,1,3))
    plt.plot(fbins/1000, shot_noise/gain, ':', label='shot')
    plt.plot(tnr_f/1000, tnr_v2[:,i]*1.e-12, 'o', \
        markerfacecolor='k', label = 'TNR', ms = 2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([4, 256])
    plt.ylim([1e-17, 1e-12])
    plt.xlabel(r'$f\:[\mathrm{kHz}]$')
    plt.ylabel(r'$V_r^2\:[\mathrm{V^2\:Hz^{-1}}]$')
    plt.title(tnr_times[i].strftime("%Y-%m-%d/%H:%M:%S"))
    plt.legend(loc='best', fontsize=9, frameon = False)
    
    param_string = '$n_{\mathrm{e}} =$ \n$n =$\n$t =$\n$t_{\mathrm{c}} =$\n$t_{\mathrm{p}} =$\n$V_{\mathrm{sw}} =$'

    units_string = "$\mathrm{cm^{-3}}$ \n$DELETEME$ \n$DELETEME$ \n$\mathrm{eV}$ \n$\mathrm{eV}$ \n$\mathrm{kmINSERTSLASHs}$"

    plt.annotate(param_string, xy=(7, 2.e-17), ha = 'right')
        
    values_string = (
        "${0:7.2f}$\n${1:7.3f}$\n${2:7.2f}$\n${3:7.2f}$\n${4:7.2f}$"
        "\n${5:7.1f}$".format(ne[i],n[i],t[i],tc[i],tp[i],vsw[i]/1.e3))
        
    plt.annotate(values_string, xy=(12, 2.e-17), ha = 'right')

    plt.annotate(units_string, xy =(12.5, 2.e-17), ha='left')

    fig.savefig(qtn_plot + '.pdf')

#%%

strt_ind = bisect.bisect_right(tnr_dts,strt_t)
stop_ind = bisect.bisect_left(tnr_dts,stop_t)

tnr_v2_interval = np.square(tnr_dat['y'][:,strt_ind:stop_ind])
tnr_dt_interval = tnr_dts[strt_ind:stop_ind]

nn_interval_inds = []
for tnr_dt in tnr_dt_interval:
    nn_interval_inds.append(nn_dts.index(min(nn_dts, \
        key=lambda dt:abs(dt-tnr_dt))))

nn_dt_interval = [nn_dts[ind] for ind in nn_interval_inds]
nn_ne_interval = np.asarray([nn_ne[ind] for ind in nn_interval_inds])
nn_fp_interval = np.sqrt(nn_ne_interval) * 9.e3

tnr_fpind_interval = tnr_f.searchsorted(nn_fp_interval)

tnr_dt_positions = []
tnr_v2_min = []
gain_min = []
tnr_v2_min_f = []
pnoise_v2_min = []
enoise_ratio = []
      
#tnr_dt_pre_ind = bisect.bisect_right(tnr_dt_interval,exhaust_bounds[0])
#tnr_dt_pst_ind = bisect.bisect_right(tnr_dt_interval,exhaust_bounds[1])

restored_file = ''

annot_str = "P   ne   te    tnr_min   nn_fp    file    ix  tnr_minf  pnoise    gain tnr_dt    en0   enf    plat" 

for i, ind in enumerate(tnr_fpind_interval):
    
    #print(min([abs(x - tnr_dt_interval[i]) for x in tnr_lfnoise_dts]))
    tnr_dt_position = -1
    
    dt_i = tnr_dt_interval[i]   
    
    if dt_i > exhaust_bounds[0][0] and dt_i < exhaust_bounds[0][1]:
        tnr_dt_position = 0
    if dt_i > exhaust_bounds[1][0] and dt_i < exhaust_bounds[1][1]:
        tnr_dt_position = 1
    if dt_i > exhaust_bounds[2][0] and dt_i < exhaust_bounds[2][1]:
        tnr_dt_position = 2
        
    tnr_dt_positions.append(tnr_dt_position)    

    tnr_lfnoise_delta = (
        min([abs(x - tnr_dt_interval[i]) for x in tnr_lfnoise_dts])
        )
        
    if (tnr_dt_position > -1) and tnr_lfnoise_delta != datetime.timedelta(0):

        snippet = tnr_v2_interval[ind-15:ind+3,i]
        snippet_f = tnr_f[ind-15:ind+3]
       
        tnr_v2_min_ind_i = np.argmin(snippet)
        tnr_v2_min_i = snippet[tnr_v2_min_ind_i]
        #tnr_v2_min_i = (np.average(snippet[8:11]))
        
        tnr_v2_min.append(tnr_v2_min_i)
    
        tnr_v2_min_f_i = snippet_f[tnr_v2_min_ind_i]
        
        tnr_v2_min_f.append(tnr_v2_min_f_i)
                
        ne_norm = ne[tnr_dt_position]   
        tc_norm = tc[tnr_dt_position]
        
        qtn_fi_str = tnr_times[tnr_dt_position].strftime("%Y%m%d_%H%M%S")
        qtn_file = qtn_dir + 'data/qtn_spec' + qtn_fi_str
       
        if os.path.isfile(qtn_file + '.npz'):
            if restored_file != qtn_file:
                print('Restoring ', qtn_file)    
                print(annot_str)
    
                qtn_saved_data = np.load(qtn_file + '.npz')
                proton_noise = qtn_saved_data['p_noise']   
                electron_noise = qtn_saved_data['e_noise']            
                restored_file = qtn_file
            
            gain_interp = interp1d(np.log10(fbins),
                                   np.log10([float(x) for x in gain]))               
    
            gain_int_f = np.log10(tnr_v2_min_f_i)
    
            gain_int_val = np.power(10.0, gain_interp(gain_int_f))
    
            gain_min.append(gain_int_val)
    
            pnoise_interp = interp1d(np.log10(fbins), 
                                     np.log10([float(x) for x in proton_noise]))        
                            
                    
            pnoise_int_f = np.log10(tnr_v2_min_f_i)
                    
            pnoise_int_val = np.power(10.0, pnoise_interp(pnoise_int_f)) * 1e12        
                    
            pnoise_v2_min.append(pnoise_int_val)
    
            enoise_interp = interp1d(np.log10(fbins),
                                     np.log10([float(x) for x in electron_noise]))        
            
            #enoise_int_f = np.log10([fbins[0], tnr_v2_min_f_i])        
            #enoise_int_f = np.log10([tnr_v2_min_f_i/2., tnr_v2_min_f_i])    
            enoise_int_f = np.log10(9.e3 * np.sqrt(ne_norm) * np.asarray([0.5, tnr_v2_min_f_i/nn_fp_interval[i]]))    
            
            enoise_int_vals = np.power(10.0, enoise_interp(enoise_int_f)) * 1e12        
                    
            enoise_ratio_i = enoise_int_vals[0]/enoise_int_vals[1]
            
            enoise_ratio.append(enoise_ratio_i)       
            
        else:
    
            pnoise_v2_min.append(np.nan)
            pnoise_int_val = np.nan
            enoise_ratio.append(np.nan)
            gain_min.append(np.nan)
            
        tempstr = (
            "{0:1d} {1:5.1f} {2:5.1f} {3:9.2e} "
            "{4:9.2e} {5:s} {6:3d} {7:9.2e} {8:9.2e} {9:5.2f}"
            .format(tnr_dt_position, ne_norm, tc_norm, tnr_v2_min_i,
                    nn_fp_interval[i], os.path.basename(qtn_file)[-6:], ind, 
                    tnr_v2_min_f_i,pnoise_int_val,gain_int_val)
            + " " + tnr_dt_interval[i].strftime("%H:%M:%S") + " "
            + "".join('{0:6.3f}'.format(x) for x in enoise_int_vals * 1.e2 / gain_int_val)
            + " {0:6.3f}".format((tnr_v2_min_i *gain_int_val - pnoise_int_val) 
                * 1.e2 * enoise_int_vals[0]/enoise_int_vals[1]))
    
    
        print(tempstr)
    
        #fig = plt.figure(figsize=[6, 6])
    
        #matplotlib.rc('text', usetex=False)
        #plt.plot(snippet_f, snippet,'-')
        #plt.plot(tnr_v2_min_f_i, tnr_v2_min_i, 'o')
        #plt.ylim([0., 0.1])
        #plt.xlim([10000., 60000.])
        #plt.annotate("\n".join(tempstr.split()), xy = (20000,0.04))
        #plt.annotate("\n".join(annot_str.split()), xy = (12000,0.04))
    else:
        print(tnr_dt_interval[i], "Out of bounds or LF Noise")
        tnr_v2_min.append(np.nan)
        pnoise_v2_min.append(np.nan)
        pnoise_int_val = np.nan
        enoise_ratio.append(np.nan)
        gain_min.append(np.nan)


tnr_dt_positions = np.asarray(tnr_dt_positions)
tnr_v2_min = np.asarray(tnr_v2_min)
gain_min = np.asarray(gain_min)
pnoise_v2_min = np.asarray(pnoise_v2_min)
enoise_ratio = np.asarray(enoise_ratio)

plateau_v2 = (tnr_v2_min * gain_min - pnoise_v2_min) * enoise_ratio

#%% Interpolations on a 2d n-T plot

ne_2d_all = np.arange(4., 31., 1.)

tc_2d_all = np.arange(4., 25., 1.)

tc_2d_grid, ne_2d_grid = np.meshgrid(tc_2d_all, ne_2d_all)

v_plateau_2d = np.zeros(ne_2d_grid.shape)

plateau_file = qtn_dir + 'data/qtn_plateau'

if os.path.isfile(plateau_file + '.npz'):
    plateau_saved_data = np.load(plateau_file + '.npz')
    v_plateau_2d = plateau_saved_data['v_plateau_2d']
else:
    for i, ne_2d in enumerate(ne_2d_all):
        for j, tc_2d in enumerate(tc_2d_all):
            n_2d = 0.05
            t_2d = 4.
            vsw_2d = 400000.
            tp_2d = 20.
            fp_2d = 9.e3 * np.sqrt(ne_2d)
            v_plateau_2d[i,j] = p.electron_noise(fp_2d/2.,ne_2d,n_2d,t_2d,
                                                 tp_2d,tc_2d,vsw_2d)
            outstr = (
                "{0:5.1f}, {1:5.1f}, {2:9.2e}"
                .format(ne_2d,tc_2d,v_plateau_2d[i,j]))
            print(outstr)
    np.savez(plateau_file, v_plateau_2d = v_plateau_2d)
       
v_plateau_interp = RectBivariateSpline(ne_2d_all, tc_2d_all, v_plateau_2d*1.e15)

def t_plateau(n, v_plateau, v_plateau_interp):
    search_t = np.arange(4,22,0.01)
    
    search_v = v_plateau_interp(n,search_t)[0]

    closest_v_ind = (np.abs(search_v -v_plateau)).argmin()

    return search_t[closest_v_ind]
    
n_pl_test = 15

v_plateau = 16.5

print(t_plateau(n_pl_test,v_plateau,v_plateau_interp))

#%%

t_pl_all = []

for i, ne_i in enumerate(nn_ne_interval):
    v2_i = plateau_v2[i] * 1.e3
    if np.isfinite(plateau_v2[i]):
        t_pl_all.append(t_plateau(ne_i,v2_i,v_plateau_interp))
    else:
        t_pl_all.append(np.nan)
        
t_pl_all = np.asarray(t_pl_all)
#plt.figure()
#plt.plot(t_pl_all)
fig = plt.figure(figsize=[5, 4])

CS = plt.contour(ne_2d_grid, tc_2d_grid, v_plateau_2d*1.e15, colors = 'k')

manual_locations = [(27,7), (27,9),(27,10),(27,15),(27,18),(25,22)]
plt.clabel(CS, inline=1,fontsize=10, manual = manual_locations,fmt = '%5.1f')
for i in [0,1,2]:
    ind, = np.where(tnr_dt_positions == i)    
    plt.plot(nn_ne_interval[ind], t_pl_all[ind], 'o')
#plt.plot(ne, tc, 'x')
plt.ylim([5, 20])
plt.xlabel(r'$n_e\:[\mathrm{cm^{-3}]$')
plt.ylabel(r'$T_c\:[\mathrm{eV}]$')

plt.title('Electron thermal noise level $V^2\:[10^{-15}\:\mathrm{V^2Hz^{-1}}]$ \n at flat region of thermal noise spectrum ($f/f_p = 0.5$)')

arrow_dict = dict(arrowstyle="->",
                  fc="0.45", ec="0.45", connectionstyle="arc3,rad = -0.3")

plt.annotate('After exhaust', 
             xy = (8,9), 
             xytext = (9,12),
             arrowprops=arrow_dict)

arrow_dict['connectionstyle'] = "arc3, rad = 0.3"

plt.annotate('Before exhaust', 
             xy = (15.75,9.25), 
             xytext = (14,14),
             arrowprops=arrow_dict)

plt.annotate('Within exhaust', 
             xy = (22,12), 
             xytext = (18,18),
             arrowprops=arrow_dict)

fig.savefig(qtn_dir + '/plots_20070507/plateau_2d.pdf')
    
#plt.figure()
#plt.plot(t_pl_all, '-o')

tnr_dt_unix = [(x - datetime.datetime(1970,1,1)).total_seconds() for x in tnr_dt_interval]

np.savetxt(qtn_dir + 'data/t_pl_all.txt', 
           np.nan_to_num(np.transpose(np.asarray([tnr_dt_unix,nn_fp_interval,t_pl_all]))), 
            delimiter = ',')