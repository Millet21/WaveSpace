#%%
from WaveSpace.WaveAnalysis import WaveActivity as wa
from WaveSpace.Utils import ImportHelpers
import numpy as np

#%%
timeseries = ImportHelpers.load_wavedata_object("./ExampleData/WaveData_EMD")
print(timeseries)

# %%
freqRange = (35,40)
freqlist = np.arange(freqRange[0], freqRange[1], 2)

wa.find_wave_activity(timeseries, freqlist, dataBucket="AnalyticSignalSubset", nBases=3)
print(timeseries)