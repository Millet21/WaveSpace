#%%
from WaveSpace.Decomposition import FrequencyCluster
from WaveSpace.Utils import ImportHelpers
import numpy as np

#%%
timeseries = ImportHelpers.load_wavedata_object("./ExampleData/WaveData_EMD")
print(timeseries)

# %%
freqRange = (15,30)
FrequencyCluster.get_frequency_cluster(timeseries, freqList=freqRange)
print(timeseries)