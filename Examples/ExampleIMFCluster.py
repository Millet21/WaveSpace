# # Add the project root directory to the Python path when working with source code, 
# # not necessary when package is installed
# import sys
# import os
# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, path )
# print(path)
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