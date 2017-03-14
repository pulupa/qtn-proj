#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:14:02 2017

@author: pulupa
"""

import numpy as np
from datetime import datetime
import os.path
import sys
import matplotlib.pyplot as plt

qtn_code_dir = '/Users/pulupa/Documents/qtn-proj-rxn-fork/'
sys.path.append(qtn_code_dir)

from qtn.bimax import BiMax

def datestr_to_datetime(date_string):
    """Convert a date string to a datetime object
    Works for formats '%Y-%m-%d/%H:%M:%S' or '%Y-%m-%d/%H:%M:%S.%f'
    Works for a single string of an iterable list
    
    >>> datestr_to_datetime("2016-01-01/00:00:00")
    datetime.datetime(2016, 1, 1, 0, 0)
    """
    
    if isinstance(date_string,(np.generic)):
        date_string = date_string.decode("utf-8")
    
    if not isinstance(date_string, str):
        parsed_datetimes = [datestr_to_datetime(ds) for ds in date_string]
        return(parsed_datetimes)

    try:
        parsed_datetime = datetime.strptime(date_string,
                                            '%Y-%m-%d/%H:%M:%S.%f')
    except ValueError:
        parsed_datetime = datetime.strptime(date_string,
                                            '%Y-%m-%d/%H:%M:%S')
    return(parsed_datetime)


def find_nearest_time_ind(index_dts, search_dts):
    """Find nearest datetimes for given index array in another search array.
    Both index_dts and search_dts are arrays of datetimes.
    For each datetime element of (n_i) index_dts, find the nearest element 
    (smallest timedelta) in the (n_s) search_dts array.  
    Return (length n_s) list of index integers corresponding the index of
    the closest element in the index_dts array, and (length n_s) list of 
    timedelta values.
    
    >>> from datetime import datetime, timedelta
    >>> index_dts = [datetime(2004,1,3) + timedelta(i) for i in range(5)]
    >>> search_dts = [datetime(2004,1,1) + timedelta(3*i) for i in range(4)]
    >>> near_ind, near_dts = find_nearest_time_ind(index_dts, search_dts)
    
    >>> print(index_dts)
    [datetime.datetime(2004, 1, 3, 0, 0),
     datetime.datetime(2004, 1, 4, 0, 0),
     datetime.datetime(2004, 1, 5, 0, 0),
     datetime.datetime(2004, 1, 6, 0, 0),
     datetime.datetime(2004, 1, 7, 0, 0)]   
    >>> print(search_dts)
    [datetime.datetime(2004, 1, 1, 0, 0),
     datetime.datetime(2004, 1, 4, 0, 0),
     datetime.datetime(2004, 1, 7, 0, 0),
     datetime.datetime(2004, 1, 10, 0, 0)]
    >>> print(near_ind)
    [0, 1, 4, 4]    
    >>> print(near_dts)
    [datetime.timedelta(2), 
     datetime.timedelta(0), 
     datetime.timedelta(0),
     datetime.timedelta(-3)]
    """    
    timedeltas = [min(index_dts, 
                      key = lambda dt:abs(dt - search_dt)) - search_dt \
                                for search_dt in search_dts]
    
    ind = [index_dts.index(min(index_dts, 
                               key = lambda dt:abs(dt - search_dt)))
                                for search_dt in search_dts]
    
    return ind, timedeltas

def load_sw_evdf_params(qtn_dir, date, evdf_times):
    """Load solar wind eVDF parameters from a text file
    """
    
    evdf_txt = qtn_dir + date + '/' + date + '_evdf_data.txt'
    
    evdf_dat = np.genfromtxt(evdf_txt, dtype = None, names = True)

    evdf_all_times = datestr_to_datetime(evdf_dat['Time'])
    
    evdf_all_inds = find_nearest_time_ind(evdf_all_times, evdf_times)[0]
    
    vsw = evdf_dat['v_sw'][evdf_all_inds] * 1.e3
    
    nc = evdf_dat['n_core'][evdf_all_inds]
    nh = evdf_dat['n_halo'][evdf_all_inds]
    
    tc = ((evdf_dat['tper_core']*2 + evdf_dat['tpar_core'])/3)[evdf_all_inds]
    th = ((evdf_dat['tper_halo']*2 + evdf_dat['tpar_halo'])/3)[evdf_all_inds]
    
    ne = nc + nh
    n = nh / nc
    t = th/tc
    
    tp = evdf_dat['t_p']
    fpe = [8980. * np.sqrt(ne)]
    
    return vsw, nc, nh, tc, th, ne, n, t, tp, fpe

def tnr_spec(tnr_plot_time, qtn_dir, date, 
             ant_len, ant_rad, base_cap,
             fbins, ne, n, t, tp, tc, vsw, tnr_f, tnr_plot_v2):
    
    time_str = tnr_plot_time.strftime("%Y%m%d_%H%M%S")

    qtn_file = qtn_dir + date + '/qtn_spec_' + time_str
    
    print(qtn_file + '.npz')
    
    if not os.path.isfile(qtn_file + '.npz'):

        p = BiMax(ant_len, ant_rad, base_cap)

        proton_noise = \
        np.array([p.proton(f,ne,n,t,tp,tc,vsw) \
        for f in fbins])
        
        electron_noise = \
        np.array([p.electron_noise(f,ne,n,t,tp,tc,vsw) \
        for f in fbins])
        
        gain_shot = \
        np.array([np.array(p.gain_shot(f,ne,n,t,tp,tc,vsw)) \
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
    plt.plot(tnr_f/1000, tnr_plot_v2*1.e-12, 'o', \
        markerfacecolor='k', label = 'TNR', ms = 2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([4, 256])
    plt.ylim([1e-17, 1e-12])
    plt.xlabel(r'$f\:[\mathrm{kHz}]$')
    plt.ylabel(r'$V_r^2\:[\mathrm{V^2\:Hz^{-1}}]$')
    plt.title(tnr_plot_time.strftime("%Y-%m-%d/%H:%M:%S"))
    plt.legend(loc='best', fontsize=9, frameon = False)
    
    param_string = ['$n_{\mathrm{e}} =$',
                    '$n =$', '$t =$', 
                    '$t_{\mathrm{c}} =$',
                    '$t_{\mathrm{p}} =$',
                    '$V_{\mathrm{sw}} =$']
    

    for j, txt in enumerate(param_string):
        plt.annotate(txt, xy = (7, 6.e-16/np.power(2,j)), ha = 'right')

    values_string = ['${0:7.2f}$'.format(ne),
                     '${0:7.3f}$'.format(n),
                     '${0:7.2f}$'.format(t),
                     '${0:7.2f}$'.format(tc),
                     '${0:7.2f}$'.format(tp),
                     '${0:7.1f}$'.format(vsw/1.e3)]

    for j, txt in enumerate(values_string):
        plt.annotate(txt, xy = (12, 6.e-16/np.power(2,j)), ha = 'right')

    units_string = ["$\mathrm{cm^{-3}}$",
                    "$ $",
                    "$ $",
                    "$\mathrm{eV}$",
                    "$\mathrm{eV}$",
                    "$\mathrm{km/s}$"]

    for j, txt in enumerate(units_string):
        plt.annotate(txt, xy = (12.5, 6.e-16/np.power(2,j)), ha = 'left')

    #plt.annotate(param_string, xy=(7, 2.e-17), ha = 'right')
        
#    values_string = (
#        "${0:7.2f}$\n${1:7.3f}$\n${2:7.2f}$\n${3:7.2f}$\n${4:7.2f}$"
#        "\n${5:7.1f}$".format(ne[i],n[i],t[i],tc[i],tp[i],vsw[i]/1.e3))
#        
#    plt.annotate(values_string, xy=(12, 2.e-17), ha = 'right')

#    plt.annotate(units_string, xy =(12.5, 2.e-17), ha='left')

    fig.savefig(qtn_file + '.pdf')

   
    return proton_noise, electron_noise, shot_noise, gain, qtn_file
