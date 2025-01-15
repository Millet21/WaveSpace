from WaveSpace.WaveAnalysis import ClusterGradient
from WaveSpace.Utils import ImportHelpers

#%%
waveData = ImportHelpers.load_wavedata_object("./ExampleData/WaveData_EMD")
print(waveData)
ClusterGradient.perform_cluster_gradient(waveData)
print(waveData)