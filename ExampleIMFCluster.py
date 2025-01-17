#%%
from WaveSpace.Decomposition import IMFCluster
from WaveSpace.Utils import ImportHelpers
import numpy as np

#%%
timeseries = ImportHelpers.load_wavedata_object("ExampleData/WaveData_EMD")
print(timeseries)

# %%
highPass = 3
lowPass = 45
IMFCluster.cluster_imfs(timeseries,highPass,lowPass)
print(timeseries)