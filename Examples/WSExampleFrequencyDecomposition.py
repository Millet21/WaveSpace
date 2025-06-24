#%%
import sys
import os

# Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )
print(path)

from WaveSpace.Simulation import SimulationFuns
from WaveSpace.PlottingHelpers import Plotting
from WaveSpace.Utils import HelperFuns as hf
from WaveSpace.Utils import ImportHelpers
from WaveSpace.Preprocessing import Filter as filt


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

#%% Load some simulated data
dataPath  = os.path.join(path, "Examples/ExampleData/Output") 
waveData = ImportHelpers.load_wavedata_object(dataPath + "/SimulatedData")
#waves were simulated at 10Hz. We can confirm that by plotting the PSD (for each trial-type)
trialInfo = waveData.get_trialInfo() #this contains the condition name for each trial
unique_conds = np.unique(trialInfo)
for cond in unique_conds:
    trial_indices = [i for i, info in enumerate(waveData.get_trialInfo()) if info == cond]
    f, psd = welch(waveData.get_data("SimulatedData")[trial_indices], fs=waveData.get_sample_rate(), nperseg=256)
    #average over trials and grid positions (channels)
    psd = np.mean(psd, axis=(0,1,2))
    plt.semilogy(f, psd)
    plt.title(f"Power Spectral Density - {cond}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.axvline(10, color='red', linestyle=':', linewidth=2)  
    plt.xlim(0, 50)
    plt.grid()
    plt.show()

#%% Filter Hilbert approach
# we filter the data narrowly around our frequency of interest (10Hz) and then apply the Hilbert transform to get the analytic signal.
# Note that this only makes sense if we **already know** that there is a narrowband oscillation at the frequency of interest. 
# To demonstrate this, we will filter the data at 15Hz as well.

for freqInd, freq in enumerate([10, 15]):  
    filt.filter_narrowband(waveData, dataBucketName = "SimulatedData", LowCutOff=freq-1, HighCutOff=freq+1, type = "FIR", order=100, causal=False)
    waveData.DataBuckets[str(freq)] =  waveData.DataBuckets.pop("NBFiltered")


temp = np.stack((waveData.DataBuckets["10"].get_data(), waveData.DataBuckets["15"].get_data()),axis=0)
waveData.add_data_bucket(wd.DataBucket(temp, "NBFiltered", "freq_trl_chan_time", waveData.get_channel_names()))

# get complex timeseries
hilb.apply_hilbert(waveData, dataBucketName = "NBFiltered")