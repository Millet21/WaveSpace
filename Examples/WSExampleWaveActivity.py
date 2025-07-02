#%%
# Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )
print(path)

#%%
from WaveSpace.WaveAnalysis import WaveActivity as wa
from WaveSpace.Utils import ImportHelpers
import numpy as np

#%%

waveData = ImportHelpers.load_wavedata_object("ExampleData/Output/ComplexData")

# %%
freqRange = (5,20)
freqlist = np.arange(freqRange[0], freqRange[1], 2)

wa.find_wave_activity(waveData, freqlist, dataBucketName="AnalyticSignal", nBases=3)
