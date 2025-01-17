#%%
from WaveSpace.Utils import ImportHelpers
from WaveSpace.Decomposition import Morlet as mw
import numpy as np
import matplotlib.pyplot as plt

#%%
timeSeries= ImportHelpers.load_wavedata_object("./ExampleData\WaveData_SIM_planewaves_onset300_highSNR")
print(timeSeries)

frequencies = np.logspace(np.log10(5), np.log10(40), num=10, base=10.0)
#%% Wavelet transform (freq-domain) 
mw.freq_domain_wavelet(timeSeries, "SimulatedData", frequencies)

print(timeSeries)

#%% Wavelet transform (time-domain) 
mw.convolution_wavelet(timeSeries, "SimulatedData", frequencies ,N_cycles=2)
# %%