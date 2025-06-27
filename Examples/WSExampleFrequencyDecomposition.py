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
from WaveSpace.Decomposition import Hilbert as hilb
from WaveSpace.Decomposition import EMD as emd
from WaveSpace.Utils import WaveData as wd
from WaveSpace.Decomposition import GenPhase 

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Load some simulated data
dataPath  = os.path.join(path, "Examples/ExampleData/Output") 
waveData = ImportHelpers.load_wavedata_object(dataPath + "/SimulatedData")

#%%
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
waveData.add_data_bucket(wd.DataBucket(temp, "NBFiltered", "freq_trl_posx_posy_time", waveData.get_channel_names()))
# get complex timeseries
hilb.apply_hilbert(waveData, dataBucketName = "NBFiltered")

#plot. Try both frequencies and see for which one the phase makes sense 
analytic_signal = waveData.DataBuckets["AnalyticSignal"].get_data()[0,0,18,19,:] #dimord is freq_trl_posx_posy_time
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# real part and envelope
axs[0].plot(waveData.get_time(), np.real(analytic_signal), label='Real part')
axs[0].plot(waveData.get_time(), np.abs(analytic_signal), label='Envelope', linestyle='--')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Real part and Envelope of Analytic Signal')
axs[0].legend()
axs[0].grid()

# phase
axs[1].plot(waveData.get_time(), np.angle(analytic_signal), color='tab:orange')
axs[1].set_ylabel('Phase (radians)')
axs[1].set_xlabel('Time (s)')
axs[1].set_title('Phase of Analytic Signal')
axs[1].grid()

plt.tight_layout()
plt.show()

waveData.save_to_file(os.path.join(dataPath, "ComplexData"))
#%% We can also do alternative decompositions
# Generalised phase (adapted from # https://github.com/mullerlab/generalized-phase)
lowerCutOff = 1
higherCutOff = 40
filt.filter_broadband(waveData, "SimulatedData", lowerCutOff, higherCutOff, 5)
GenPhase.generalized_phase(waveData, "BBFiltered")
#plot
complexSignal = waveData.DataBuckets["ComplexPhaseData"].get_data()[0,18,19,:] #dimord is freq_trl_posx_posy_time
bbFiltered = waveData.DataBuckets["BBFiltered"].get_data()[0,18,19,:]

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# real part and envelope
axs[0].plot(waveData.get_time(), np.real(complexSignal), label='Real part')
axs[0].plot(waveData.get_time(), bbFiltered, label='Envelope', linestyle='--')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Real part and Envelope of Analytic Signal')
axs[0].legend()
axs[0].grid()
#note that we do not plot the real and the envelop of an analytic signal here (like we did above). 
# This is because we did not narrowband filter before genPhase and the envelope of the real signal after genPhase 
# would look very weird 
# phase
axs[1].plot(waveData.get_time(), np.angle(complexSignal), color='tab:orange')
axs[1].set_ylabel('Phase (radians)')
axs[1].set_xlabel('Time (s)')
axs[1].set_title('Phase of Analytic Signal')
axs[1].grid()

plt.tight_layout()
plt.show()

#alternative plot closer to the figure shown in # https://github.com/mullerlab/generalized-phase
from matplotlib.collections import LineCollection
from scipy.io import loadmat
mat = loadmat(os.path.join(dataPath, "exampleChannel.mat"))
data = mat['x']
time = waveData.get_time()
xw = np.real(bbFiltered)
xgp = complexSignal
phase = np.angle(xgp)
fig = plt.figure(figsize=(12.5, 4.2))
ax1 = fig.add_axes([0.08, 0.15, 0.7, 0.75])  

ax1.plot(time, xw, linewidth=4, color='k', label='wideband signal')

# Colored phase line
points = np.array([time, np.real(xgp)]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(-np.pi, np.pi)
lc = LineCollection(segments, cmap='hsv', norm=norm)
lc.set_array(phase)
lc.set_linewidth(5)
ax1.add_collection(lc)

# Normal axes
ax1.set_xlim([time[0], time[-1]])
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude (a.u.)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2 = fig.add_axes([0.2116, 0.6976, 0.0884, 0.2000], polar=True)
theta = np.linspace(-np.pi, np.pi, 100)
for i in range(len(theta)-1):
    ax2.plot(theta[i:i+2], [1, 1], color=plt.cm.hsv(norm(theta[i])), linewidth=6)
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_axis_off()

# %% Empirical mode decomposition (EMD) 
# If we cannot expect the signal to be well behaved for FFT based approaches, we can use EMD
# note that this is A LOT slower than Filter + Hilbert

#We cut down the data to a small region to speed up the example
waveData.DataBuckets["SimulatedData"].set_data(waveData.get_data("SimulatedData")[0:2,10:14,10:14,:], "trl_posx_posy_time")

emd.EMD(waveData, 
        siftType = 'masked_sift',
        nIMFs=7, 
        dataBucketName="SimulatedData", 
        noiseVar = 0.05, 
        n_noiseChans = 10, 
        ndir=None, 
        stp_crit ='stop', 
        sd=0.075, 
        sd2=0.75, 
        tol=0.075,
        stp_cnt=2)

#plot imfs
TrialOfInterest = 0
SelectedChannel = (1,1)
IMFOfInterest = 4
dataInds = (slice(None), TrialOfInterest, SelectedChannel[0], SelectedChannel[1])
Plotting.plot_imfs(waveData, dataInds, IMFOfInterest)


