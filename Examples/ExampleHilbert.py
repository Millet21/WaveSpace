# # Add the project root directory to the Python path when working with source code, 
# # not necessary when package is installed
# import sys
# import os
# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, path )
# print(path)
# %%
from WaveSpace.Utils import ImportHelpers
from WaveSpace.Preprocessing import Filter as preproc
from WaveSpace.Decomposition import Hilbert

#%%
timeSeries= ImportHelpers.load_wavedata_object("ExampleData/WaveData_SIM_planewaves_onset300_lowSNR")
print(timeSeries)

#%% 
LowCutOff = 12
HighCutOff = 18

preproc.filter_narrowband(timeSeries,"SimulatedData",LowCutOff=LowCutOff, HighCutOff=HighCutOff)
print(timeSeries)

Hilbert.apply_hilbert(timeSeries, "NBFiltered" )
print(timeSeries)
# %%