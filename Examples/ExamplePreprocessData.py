
#%%
from WaveSpace.Utils import ImportHelpers
from WaveSpace.Preprocessing import Filter as preproc
import matplotlib.pyplot as plt
import numpy as np

#%% Example, load data from disk and apply a narrowband filter
LowCutOff = 4
HighCutOff = 10
data= ImportHelpers.load_wavedata_object("ExampleData/WaveData_SIM_planewaves_onset300_highSNR")

#Plot an example channel
thischan=0
thistrial = 0
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(data.get_time(),data.get_active_data()[thistrial,thischan,:])
plt.title('example channel ' + str(thischan) + ' raw')

#Filter
preproc.filter_narrowband(data,"SimulatedData",  LowCutOff, HighCutOff)
print(data)
#plot same example channel filtered
plt.subplot(2,1,2)
plt.plot(data.get_time(),data.get_data("NBFiltered")[thistrial,thischan,:])
plt.title('example channel ' + str(thischan) + 
        ' filtered ' + str(LowCutOff) + ' to ' + str(HighCutOff))

#Note the filter artefact on the edges, we can cut those out before further analysis
data.crop_data(0.5)


#%% Example, load data from disk and apply a notch plus a broadband filter
data= ImportHelpers.load_wavedata_object("ExampleData/WaveData_SIM_planewaves_onset300_highSNR")

#Plot an example channel
thischan=0
thistrial = 0
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(data.get_time(),data.get_active_data()[thistrial,thischan,:])
plt.title('example channel ' + str(thischan) + ' raw')

#Filter
lineNoiseFreq = 50
LowCutOff = 1
HighCutOff = 125
preproc.filter_notch(data, "SimulatedData", lineNoiseFreq)
preproc.filter_broadband(data,"SimulatedData", LowCutOff, HighCutOff)

#plot same example channel filtered
plt.subplot(2,1,2)
plt.plot(data.get_time(),data.get_active_data()[thistrial,thischan,:])
plt.title('example channel ' + str(thischan) + 
        ' filtered ' + str(LowCutOff) + ' to ' + str(HighCutOff))

#Note the filter artefact on the edges, we can cut those out before further analysis
data.crop_data(0.5)

# %%