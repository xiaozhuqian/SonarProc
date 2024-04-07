import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d

 
def gray_enhancing_statistic_fit(bs_geocorrection, gain=50.):
    n_sample = bs_geocorrection.shape[1]
    n_ping = bs_geocorrection.shape[0]
    port = np.fliplr(bs_geocorrection[:,0:n_sample//2])
    starboard = bs_geocorrection[:,(n_sample//2):]
    
    bs_dic = {'port':port, 'starboard': starboard}
    factor_dic = {'port': port[0], 'starboard':starboard[0]}
    bs_pingsec_mean_dic = {'port': port[0], 'starboard':starboard[0]}
    for (key, value) in bs_dic.items():
        bs_value = value
        bs_pingsec_mean = np.mean(bs_value, axis=0)
        bs_mean = np.mean(bs_value)
        factor = bs_mean/bs_pingsec_mean
        bs_pingsec_mean_dic[key] = bs_pingsec_mean
        factor_dic[key] = factor

    bs_pingsec_mean_dual = np.concatenate((bs_pingsec_mean_dic['port'][::-1], bs_pingsec_mean_dic['starboard']),axis=0)
    factor_dual = np.concatenate((factor_dic['port'][::-1], factor_dic['starboard']),axis=0)
    #factor_dual_smooth = savgol_filter(factor_dual, 51, 2, mode= 'nearest')
    x = np.arange(0, factor_dual.shape[0])
    y9 = poly1d(polyfit(x,factor_dual,9))
    factor_dual_smooth = y9(x)
    
    bs_corrected = bs_geocorrection*factor_dual_smooth*gain
    bs_corrected_clip = np.uint8(np.minimum(np.maximum(bs_corrected, 0), 255))

    plt.figure(figsize=(40,15))
    plt.subplot(611)
    plt.plot(np.arange(0, n_sample), bs_geocorrection[446])
    plt.title('bs')
    plt.subplot(612)
    plt.plot(np.arange(0, n_sample), bs_corrected[446])
    plt.title('bs_corrected_mul')
    plt.subplot(613)
    plt.plot(np.arange(0, n_sample), bs_corrected_clip[0])
    plt.title('bs_corrected_mul_clip')
    plt.subplot(614)
    plt.plot(np.arange(0, n_sample),bs_pingsec_mean_dual)
    plt.title('bs_pingsec_mean_plus')
    plt.subplot(615)
    plt.plot(np.arange(0, n_sample),factor_dual)
    plt.title('factor_dual')
    plt.subplot(616)
    plt.plot(np.arange(0, n_sample),factor_dual_smooth)
    plt.title('factor_dual_smooth')
    plt.show()

    return bs_corrected_clip