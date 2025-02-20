#%%
from WaveSpace.Utils import ImportHelpers
from WaveSpace.Decomposition import GenPhase 

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio

#%%
timeSeries= ImportHelpers.load_wavedata_object("ExampleData/WaveData_SIM_planewaves_onset300_highSNR")
print(timeSeries)

#%% Calculate generalized phase
GenPhase.generalized_phase(timeSeries)
print(timeSeries)

#%%
#Plot an example channel
thischan=0
thistrial = 0
plt.figure(figsize=(10,4))
plt.plot(timeSeries.get_time(),timeSeries.DataBuckets["SimulatedData"].get_data()[thistrial,thischan,:])
plt.title('Channel ' + str(thischan) + 'Raw')
plt.show()

plt.figure(figsize=(10,4))
plt.plot(timeSeries.get_time(),timeSeries.DataBuckets["ComplexPhaseData"].get_data()[thistrial,thischan,:])
plt.title('Channel ' + str(thischan) + 'Amplitude')
plt.show()

plt.figure(figsize=(10,4))
plt.plot(timeSeries.get_time(),np.angle(timeSeries.DataBuckets["ComplexPhaseData"].get_data()[thistrial,thischan,:]))
plt.title('Channel ' + str(thischan) + 'Phase Angle')
plt.show()

#Filter

lineNoiseFreq = 50
LowCutOff = 1
HighCutOff = 125
#data = preproc.filter_notch(data,lineNoiseFreq)
#data = preproc.filter_broadband(data, LowCutOff, HighCutOff)
print(timeSeries)
# %%

assert timeSeries.DataBuckets["ComplexPhaseData"].get_data().max() <= np.pi and  timeSeries.DataBuckets["ComplexPhaseData"].get_data().min() >= -np.pi 
# %%