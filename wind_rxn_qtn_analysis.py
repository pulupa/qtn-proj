# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:17:14 2015

@author: pulupa
"""
from scipy.io.idl import readsav
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import bisect
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
import json
from pprint import pprint
import os.path
import sys

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['xtick.direction'] = 'out'; rcParams['ytick.direction'] = 'out';
matplotlib.rc('text', usetex=True)
rcParams["text.latex.preview"] = True

#from qtn.bimax import BiMax

qtn_dir = '/Users/pulupa/box/reconnection/'

#%%
"""
Load the utilities programs
"""

qtn_code_dir = '/Users/pulupa/Documents/qtn-proj-rxn-fork/'

sys.path.append(qtn_code_dir)

from wind_rxn_qtn_utils import datestr_to_datetime, \
                                find_nearest_time_ind, \
                                load_sw_evdf_params, \
                                tnr_spec

    
#%%
"""
Select date for event
"""

date = '20041011'

#%%

with open(qtn_dir + date + '/' + date + '.json') as data_file:    
    json_data = json.load(data_file)

pprint(json_data)

#%%
"""
Load IDL save file data
"""

tnr_dat = readsav(qtn_dir + date + '/tnr_cal_dat_' + date + '.sav')

# Turn TNR times from IDLSAV file into datetimes

tnr_dts = []
for tnr_unix in (tnr_dat['x']):
    tnr_dts.append(datetime.utcfromtimestamp(tnr_unix))

tnr_f = tnr_dat['v']

# Turn TNR LF noise times from JSON file into datetimes

tnr_lfnoise_dts = datestr_to_datetime(json_data['tnr_lfnoise_ind'])
    
tnr_lfnoise_ind, tnr_lfnoise_delta = find_nearest_time_ind(tnr_dts, 
                                                           tnr_lfnoise_dts)
    
# Read neural network data from IDL save file

nn_dat = readsav(qtn_dir + date + '/nn_dat_' + date + '.sav')

nn_dts = []
for nn_unix in (nn_dat['tnr_fpnn_x']):
    nn_dts.append(datetime.utcfromtimestamp(nn_unix))
    
nn_ne = nn_dat['tnr_fpnn_y']

#%% 
"""
Input datetimes
"""

js_t = json_data['time_intervals']

strt_t, stop_t = datestr_to_datetime([js_t['t0_plot'],js_t['t1_plot']]) 

exhaust_bounds = datestr_to_datetime([js_t['t0_exhaust'],js_t['t1_exhaust']])

evdf_times = datestr_to_datetime([json_data['evdf_example_times']['up'],
                                  json_data['evdf_example_times']['in'],
                                  json_data['evdf_example_times']['dn']])

t_up = datestr_to_datetime([js_t['t0_up'],js_t['t1_up']])
t_in = datestr_to_datetime([js_t['t0_in'],js_t['t1_in']])
t_dn = datestr_to_datetime([js_t['t0_dn'],js_t['t1_dn']])

#%%
"""
Get the TNR and NN data for the three selected eVDF plot times
"""

tnr_plot_inds, tnr_plot_timedeltas = find_nearest_time_ind(tnr_dts, evdf_times)

tnr_plot_v2 = np.square(tnr_dat['y'][:,tnr_plot_inds])

tnr_plot_times = [tnr_dts[ind] for ind in tnr_plot_inds]

nn_evdf_inds, nn_evdf_timedeltas = find_nearest_time_ind(nn_dts, evdf_times)

nn_ne_evdfs = nn_ne[nn_evdf_inds]

nn_time_evdfs = [nn_dts[ind] for ind in nn_evdf_inds]

#%%
"""
Wind antenna parameters
"""
ant_len = 30.      # m (monopole) 
ant_rad = 1.9e-4   # m
base_cap = 20e-12  # Farad
fbins = np.array([4000*1.0445**i for i in range(96)])

#%%
"""
Load the solar wind parameters
"""

vsw, nc, nh, tc, th, ne, n, t, tp, fpe = \
    load_sw_evdf_params(qtn_dir, date, evdf_times)

#%%
"""
Noise curve calculations (if needed) and three eVDF plots
"""

for i, tnr_plot_time in enumerate(tnr_plot_times):
    
    proton_noise, electron_noise, shot_noise, gain, qtn_file = \
        tnr_spec(tnr_plot_time, qtn_dir, date, 
             ant_len, ant_rad, base_cap,
             fbins, ne[i], n[i], t[i], tp[i], tc[i], vsw[i], 
             tnr_f, tnr_plot_v2[:,i]) 

#%%
"""
Load TNR and NN data for exhaust interval
"""

# Find start and stop indices for exhaust interval

strt_ind = bisect.bisect_right(tnr_dts,strt_t)
stop_ind = bisect.bisect_left(tnr_dts,stop_t)

# Load TNR and NN data from exhaust interval

tnr_v2_interval = np.square(tnr_dat['y'][:,strt_ind:stop_ind])
tnr_dt_interval = tnr_dts[strt_ind:stop_ind]

nn_int_inds, nn_dt_interval = find_nearest_time_ind(nn_dts, tnr_dt_interval)

nn_ne_interval = np.asarray([nn_ne[ind] for ind in nn_int_inds])
nn_fp_interval = np.sqrt(nn_ne_interval) * 9.e3

# Find the f_p index in the TNR data

tnr_fpind_interval = tnr_f.searchsorted(nn_fp_interval)

#%%

tnr_dt_positions = []
tnr_v2_min = []
gain_min = []
tnr_v2_min_f = []
pnoise_v2_min = []
enoise_ratio = []
      
restored_file = ''

annot_str = "               tnrmn  nn_fp                minf      " + \
    "pnoise    gain tnr_dt    en0   enf    plat    Δlf\n" + \
            "P   ne   te    uV2/Hz  kHz   file   ix      kHz  " + \
    "pnoise    gain tnr_dt    en0   enf    plat    Δlf" 

# Create figure for plot of TNR snippets

fig = plt.figure(figsize=(5, 30))

ax = fig.add_subplot(1,1,1)

# Loop through TNR spectra

for i, ind in enumerate(tnr_fpind_interval):
    
    dt_i = tnr_dt_interval[i]   
    tnr_dt_position = -1
    
    # Select a portion of the TNR spectrum which includes the plasma peak
    # as well as the plateau below the peak.
    
    snippet = tnr_v2_interval[ind-15:ind+5,i]
    snippet_f = tnr_f[ind-15:ind+5]

    # Check for times of high LF noise

    if len(tnr_lfnoise_dts) == 0:
        tnr_lfnoise_delta = None;        
    else:
        tnr_lfnoise_delta = (
            min([abs(x - tnr_dt_interval[i]) for x in tnr_lfnoise_dts])
            )
    
    # Before exhaust: pos = 0, color = blue
    # Within exhaust: pos = 1, color = green
    # After  exhaust: pos = 2, color = red

    if dt_i > t_up[0] and dt_i < t_up[1]:
        tnr_dt_position = 0
        color = 'blue'
    if dt_i > t_in[0] and dt_i < t_in[1]:
        tnr_dt_position = 1
        color = 'green'
    if dt_i > t_dn[0] and dt_i < t_dn[1]:
        tnr_dt_position = 2
        color = 'red'
        
    tnr_dt_positions.append(tnr_dt_position)
        
    # Plot TNR data from vicinity of plasma peak
    # Overplot index of n_e with a circle on each TNR spectrum

    if (tnr_dt_position > -1):

        if tnr_lfnoise_delta < timedelta(0,3):
            color = 'black'
    
        line, = ax.plot(snippet_f,snippet*np.power(1.5,i),color = color)
        ax.plot(tnr_f[tnr_fpind_interval[i]],
                tnr_v2_interval[tnr_fpind_interval[i],i]*np.power(1.5,i),
                'o', color = color)
        ax.set_yscale('log')
        ax.set_xscale('log')
       
    if (tnr_dt_position > -1) and tnr_lfnoise_delta != timedelta(0):

        # Find the minimum V2/Hz in the plateau region.
        
        tnr_v2_min_ind_i = np.argmin(snippet)
        tnr_v2_min_i = snippet[tnr_v2_min_ind_i]
        #tnr_v2_min_i = (np.average(snippet[8:11]))
        tnr_v2_min.append(tnr_v2_min_i)

        # Find the frequency for the minimum
    
        tnr_v2_min_f_i = snippet_f[tnr_v2_min_ind_i]
        tnr_v2_min_f.append(tnr_v2_min_f_i)

        # Find the n_e and t_c for normalization (the n_e and t_c from the
        # eVDF fit in the upstream/exhaust/downstream region)

        ne_norm = ne[tnr_dt_position]   
        tc_norm = tc[tnr_dt_position]
        
        # Load the calculated QTN file for the eVDF
        
        time_str = tnr_plot_times[tnr_dt_position].strftime("%Y%m%d_%H%M%S")
        qtn_file = qtn_dir + date + '/qtn_spec_' + time_str
      
        if os.path.isfile(qtn_file + '.npz'):
            if restored_file != qtn_file:
                print('Restoring ', qtn_file)    
                print(annot_str)
    
                qtn_saved_data = np.load(qtn_file + '.npz')
                proton_noise = qtn_saved_data['p_noise']   
                electron_noise = qtn_saved_data['e_noise']            
                restored_file = qtn_file
            
            # Interpolate (in log-log space) the gain at the plateau frequency
            
            gain_interp = interp1d(np.log10(fbins),
                                   np.log10([float(x) for x in gain]))               
            gain_int_f = np.log10(tnr_v2_min_f_i)
            gain_int_val = np.power(10.0, gain_interp(gain_int_f))
            gain_min.append(gain_int_val)
    
            # Interpolate (in log-log space) proton noise at plateau frequency
    
            pnoise_interp = \
                interp1d(np.log10(fbins), 
                         np.log10([float(x) for x in proton_noise]))        
            pnoise_int_f = np.log10(tnr_v2_min_f_i)
            pnoise_int_val = np.power(10.0, pnoise_interp(pnoise_int_f)) * 1e12        
            pnoise_v2_min.append(pnoise_int_val)
            
            # The level of the electron noise plateau gives us the temperature.
            # Specifically, we interpolate to the contours on the 2D plateau 
            # map, whose contours are generated by calculating the level of 
            # the spectrum at 0.5 fp.  The minimum in the measured spectra,
            # however, does not occur precisely at 0.5 fp.  So to convert
            # the minimum (measured) to the value at 0.5 fp, we calculate the
            # electron noise spectrum both at the measured f and at exactly
            # 0.5 fp.  We then use the ratio of those calculated values to 
            # adjust the measured value.
            
            enoise_interp = \
                interp1d(np.log10(fbins),
                         np.log10([float(x) for x in electron_noise]))        
            
            # old versions
            # enoise_int_f = np.log10([fbins[0], tnr_v2_min_f_i])        
            # enoise_int_f = np.log10([tnr_v2_min_f_i/2., tnr_v2_min_f_i])    
            
            enoise_int_f = \
                np.log10(9.e3 * np.sqrt(ne_norm) * 
                         np.asarray([0.5, tnr_v2_min_f_i/nn_fp_interval[i]]))    
            
            enoise_ints = np.power(10.0, enoise_interp(enoise_int_f)) * 1e12        
                    
            enoise_ratio_i = enoise_ints[0]/enoise_ints[1]
            
            enoise_ratio.append(enoise_ratio_i)       
            
        else:
    
            pnoise_v2_min.append(np.nan)
            pnoise_int_val = np.nan
            enoise_ratio.append(np.nan)
            gain_min.append(np.nan)
        
        # Print an output summary string for each measurement
        
        tempstr = (
            "{0:1d} "
            "{1:5.1f} "
            "{2:5.1f} " 
            "{3:7.4f} "
            "{4:5.2f} " 
            "{5:s} "
            "{6:3d} "
            "{7:9.2e} "
            "{8:9.2e} " 
            "{9:5.2f} "
            "{10:s} "
            "{11:6.3f} "
            "{12:6.3f} "
            "{13:6.3f}"
            "{14:7.1f}"
            .format(tnr_dt_position,                                    #  0
                    ne_norm,                                            #  1
                    tc_norm,                                            #  2
                    tnr_v2_min_i,                                       #  3
                    nn_fp_interval[i]/1.e3,                             #  4
                    os.path.basename(qtn_file)[-6:],                    #  5
                    ind,                                                #  6
                    tnr_v2_min_f_i,                                     #  7
                    pnoise_int_val,                                     #  8
                    gain_int_val,                                       #  9
                    tnr_dt_interval[i].strftime("%H:%M:%S"),            # 10
                    enoise_ints[0] * 1.e2 / gain_int_val,               # 11
                    enoise_ints[1] * 1.e2 / gain_int_val,               # 12
                    ((tnr_v2_min_i * gain_int_val - pnoise_int_val)     # 13
                    * 1.e2 * enoise_ints[0]/enoise_ints[1]),
                    tnr_lfnoise_delta.total_seconds()))                 # 14
       
        print(tempstr)
    
    else:
        
        # Print a dummy string for each measurement contaminated by LF noise
        
        tempstr = (
            "{0:1s} "
            "{1:5.1f} "
            "{2:5.1f} " 
            "{3:9.2e} "
            "{4:9.2e} " 
            "{5:s} "
            "{6:3d} "
            "{7:9.2e} "
            "{8:9.2e} " 
            "{9:5.2f} "
            "{10:s} "
            "{11:6.3f} "
            "{12:6.3f} "
            "{13:6.3f}"
            "{14:7.1f}"
            .format('-',                                                #  0
                    0,                                                  #  1
                    0,                                                  #  2
                    0,                                                  #  3
                    0,                                                  #  4
                    '------',                                           #  5
                    0,                                                  #  6
                    0,                                                  #  7
                    0,                                                  #  8
                    0,                                                  #  9
                    tnr_dt_interval[i].strftime("%H:%M:%S"),            # 10
                    0,                                                  # 11
                    0,                                                  # 12
                    0,                                                  # 13
                    tnr_lfnoise_delta.total_seconds()))                 # 14

        print(tempstr)
        
        # Append data from each event to the lists
        
        tnr_v2_min.append(np.nan)
        pnoise_v2_min.append(np.nan)
        pnoise_int_val = np.nan
        enoise_ratio.append(np.nan)
        gain_min.append(np.nan)

#%%

# Convert data to nparrays

tnr_dt_positions = np.asarray(tnr_dt_positions)
tnr_v2_min = np.asarray(tnr_v2_min)
gain_min = np.asarray(gain_min)
pnoise_v2_min = np.asarray(pnoise_v2_min)
enoise_ratio = np.asarray(enoise_ratio)

# Calculate the V2 value of the 

plateau_v2 = (tnr_v2_min * gain_min - pnoise_v2_min) * enoise_ratio

#%% Interpolations on a 2d n-T plot

ne_2d_all = np.arange(4., 31., 1.)

tc_2d_all = np.arange(4., 25., 1.)

tc_2d_grid, ne_2d_grid = np.meshgrid(tc_2d_all, ne_2d_all)

v_plateau_2d = np.zeros(ne_2d_grid.shape)

plateau_file = qtn_dir + 'plateau_data/qtn_plateau'

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
plt.xlabel('$n_e\:[\mathrm{cm^{-3}}]$')
plt.ylabel('$T_c\:[\mathrm{eV}]$')

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

fig.savefig(qtn_dir + date + '/plateau_2d.pdf')
    
#plt.figure()
#plt.plot(t_pl_all, '-o')

tnr_dt_unix = [(x - datetime(1970,1,1)).total_seconds() for x in tnr_dt_interval]

np.savetxt(qtn_dir + date + '/tc_plateau.txt', 
           np.nan_to_num(np.transpose(np.asarray([tnr_dt_unix,nn_fp_interval,t_pl_all]))), 
            delimiter = ',')