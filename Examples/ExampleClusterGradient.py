# # Add the project root directory to the Python path when working with source code, 
# # not necessary when package is installed
# import sys
# import os
# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, path )
# print(path)

from WaveSpace.WaveAnalysis import ClusterGradient
from WaveSpace.Utils import ImportHelpers

#%%
waveData = ImportHelpers.load_wavedata_object("./ExampleData/WaveData_EMD")
print(waveData)
ClusterGradient.perform_cluster_gradient(waveData)
print(waveData)