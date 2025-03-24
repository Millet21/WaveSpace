#%%
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